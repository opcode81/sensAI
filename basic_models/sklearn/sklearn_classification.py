import logging
from typing import Sequence

import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.tree

from .sklearn_base import AbstractSkLearnVectorClassificationModel, DataFrameTransformer


log = logging.getLogger(__name__)


class SkLearnDecisionTreeVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (),
            outputTransformers: Sequence[DataFrameTransformer] = (),
            min_samples_leaf=8, random_state=42, **modelArgs):
        super().__init__(sklearn.tree.DecisionTreeClassifier,
            sklearnInputTransformer=sklearnInputTransformer, sklearnOutputTransformer=sklearnOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers,
            min_samples_leaf=min_samples_leaf, random_state=random_state, **modelArgs)


class SkLearnRandomForestVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (),
            outputTransformers: Sequence[DataFrameTransformer] = (),
            min_samples_leaf=8, random_state=42, **modelArgs):
        super().__init__(sklearn.ensemble.RandomForestClassifier,
            sklearnInputTransformer=sklearnInputTransformer, sklearnOutputTransformer=sklearnOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers,
            random_state=random_state, min_samples_leaf=min_samples_leaf, **modelArgs)


class SkLearnMLPVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (),
            outputTransformers: Sequence[DataFrameTransformer] = (),
            **modelArgs):
        super().__init__(sklearn.neural_network.MLPClassifier,
            sklearnInputTransformer=sklearnInputTransformer, sklearnOutputTransformer=sklearnOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers,
            **modelArgs)


class SkLearnMultinomialNBVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (),
            outputTransformers: Sequence[DataFrameTransformer] = (),
            **modelArgs):
        super().__init__(sklearn.naive_bayes.MultinomialNB,
            sklearnInputTransformer=sklearnInputTransformer, sklearnOutputTransformer=sklearnOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers,
            **modelArgs)
