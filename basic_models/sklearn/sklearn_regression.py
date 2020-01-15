import logging
from typing import Sequence

import sklearn.ensemble
import sklearn.linear_model
import sklearn.neighbors
import sklearn.neural_network
import sklearn.svm
import lightgbm

from .sklearn_base import AbstractSkLearnMultipleOneDimVectorRegressionModel, AbstractSkLearnMultiDimVectorRegressionModel, InvertibleDataFrameTransformer, DataFrameTransformer


log = logging.getLogger(__name__)


class SkLearnRandomForestVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    def __init__(self, sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (),
            outputTransformers: Sequence[DataFrameTransformer] = (),
            targetTransformer: InvertibleDataFrameTransformer = None,
            n_estimators=100, min_samples_leaf=10, random_state=42, **modelArgs):
        super().__init__(sklearn.ensemble.RandomForestRegressor,
            sklearnInputTransformer=sklearnInputTransformer, sklearnOutputTransformer=sklearnOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers, targetTransformer=targetTransformer,
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=random_state, **modelArgs)


class SkLearnLinearRegressionVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self, sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (),
            outputTransformers: Sequence[DataFrameTransformer] = (),
            targetTransformer: InvertibleDataFrameTransformer = None,
            **modelArgs):
        super().__init__(sklearn.linear_model.LinearRegression,
            sklearnInputTransformer=sklearnInputTransformer, sklearnOutputTransformer=sklearnOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers, targetTransformer=targetTransformer,
            **modelArgs)


class SkLearnMultiLayerPerceptronVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self, sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (),
            outputTransformers: Sequence[DataFrameTransformer] = (),
            targetTransformer: InvertibleDataFrameTransformer = None,
            random_state=42, **modelArgs):
        """
        :param sklearnInputTransformer: one of the preprocessors in sklearn.preprocessing, e.g. StandardScaler
        :param modelArgs: arguments to pass on to MLPRegressor
        """
        super().__init__(sklearn.neural_network.MLPRegressor,
            sklearnInputTransformer=sklearnInputTransformer, sklearnOutputTransformer=sklearnOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers, targetTransformer=targetTransformer,
            random_state=random_state, **modelArgs)


class SkLearnSVRVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self, sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (),
            outputTransformers: Sequence[DataFrameTransformer] = (),
            targetTransformer: InvertibleDataFrameTransformer = None,
            **modelArgs):
        super().__init__(sklearn.svm.SVR,
            sklearnInputTransformer=sklearnInputTransformer, sklearnOutputTransformer=sklearnOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers, targetTransformer=targetTransformer,
            **modelArgs)


class SkLearnLinearSVRVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self, sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (),
            outputTransformers: Sequence[DataFrameTransformer] = (),
            targetTransformer: InvertibleDataFrameTransformer = None,
            **modelArgs):
        super().__init__(sklearn.svm.LinearSVR,
            sklearnInputTransformer=sklearnInputTransformer, sklearnOutputTransformer=sklearnOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers, targetTransformer=targetTransformer,
            **modelArgs)


class SkLearnLightGBMVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self, sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (),
            outputTransformers: Sequence[DataFrameTransformer] = (),
            targetTransformer: InvertibleDataFrameTransformer = None,
            random_state=42, num_leaves=300, **modelArgs):
        super().__init__(lightgbm.sklearn.LGBMRegressor,
            sklearnInputTransformer=sklearnInputTransformer, sklearnOutputTransformer=sklearnOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers, targetTransformer=targetTransformer,
            random_state=random_state, num_leaves=num_leaves, **modelArgs)


class SkLearnGradientBoostingVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    def __init__(self, sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (),
            outputTransformers: Sequence[DataFrameTransformer] = (),
            targetTransformer: InvertibleDataFrameTransformer = None,
            random_state=42, **modelArgs):
        super().__init__(sklearn.ensemble.GradientBoostingRegressor,
            sklearnInputTransformer=sklearnInputTransformer, sklearnOutputTransformer=sklearnOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers, targetTransformer=targetTransformer,
            random_state=random_state, **modelArgs)


class SkLearnKNeighborsVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self, sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (),
            outputTransformers: Sequence[DataFrameTransformer] = (),
            targetTransformer: InvertibleDataFrameTransformer = None,
            **modelArgs):
        super().__init__(sklearn.neighbors.KNeighborsRegressor,
            sklearnInputTransformer=sklearnInputTransformer, sklearnOutputTransformer=sklearnOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers, targetTransformer=targetTransformer,
            **modelArgs)


class SkLearnExtraTreesVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    def __init__(self, sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (),
            outputTransformers: Sequence[DataFrameTransformer] = (),
            targetTransformer: InvertibleDataFrameTransformer = None,
            n_estimators=100, min_samples_leaf=10, random_state=42, **modelArgs):
        super().__init__(sklearn.ensemble.ExtraTreesRegressor,
            sklearnInputTransformer=sklearnInputTransformer, sklearnOutputTransformer=sklearnOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers, targetTransformer=targetTransformer,
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=random_state, **modelArgs)
