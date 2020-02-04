import logging
import numpy as np
import pandas as pd
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
            random_state=42, hidden_layer_sizes=(100,), activation="relu", solver="adam",
            max_iter=200, **modelArgs):
        """
        :param sklearnInputTransformer: an optional sklearn preprocessor for normalising/scaling inputs
        :param sklearnOutputTransformer: an optional sklearn preprocessor for normalising/scaling outputs
        :param inputTransformers: list of DataFrameTransformers for the transformation of inputs
        :param outputTransformers: list of DataFrameTransformers for the transformation of outputs
        :param targetTransformer: a transformer which transforms the targets (training data outputs) prior to learning the model, such
            that the model learns to predict the transformed outputs. When predicting, the inverse transformer is applied after applying
            the model, i.e. the transformation is completely transparent when applying the model.
        :param random_state: the random seed
        :param hidden_layer_sizes: the sequence of hidden layer sizes
        :param activation: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’} the activation function to use for hidden layers
            (the one used for the output layer is always 'identity')
        :param solver: {‘lbfgs’, ‘sgd’, ‘adam’} the name of the solver to apply
        :param max_iter: the number of iterations (gradient steps for L-BFGS, epochs for other solvers)
        :param modelArgs: additional arguments to pass on to MLPRegressor, see https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
        """
        super().__init__(sklearn.neural_network.MLPRegressor,
            sklearnInputTransformer=sklearnInputTransformer, sklearnOutputTransformer=sklearnOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers, targetTransformer=targetTransformer,
            # underlying model parameters
            random_state=random_state, hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter,
            **modelArgs)


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


class SkLearnLightGBMVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    log = log.getChild(__qualname__)

    def __init__(self, sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (),
            outputTransformers: Sequence[DataFrameTransformer] = (),
            targetTransformer: InvertibleDataFrameTransformer = None,
            categoricalFeatureNames: Sequence[str] = None,
            random_state=42, num_leaves=300, **modelArgs):
        """
        :param sklearnInputTransformer: an optional sklearn preprocessor for normalising/scaling inputs
        :param sklearnOutputTransformer: an optional sklearn preprocessor for normalising/scaling outputs
        :param inputTransformers: list of DataFrameTransformers for the transformation of inputs
        :param outputTransformers: list of DataFrameTransformers for the transformation of outputs
        :param targetTransformer: a transformer which transforms the targets (training data outputs) prior to learning the model, such
            that the model learns to predict the transformed outputs. When predicting, the inverse transformer is applied after applying
            the model, i.e. the transformation is completely transparent when applying the model.
        :param categoricalFeatureNames: sequence of feature names in the input data that are categorical.
            Columns that have dtype 'category' (as will be the case for categorical columns created via FeatureGenerators)
            need not be specified (should be inferred automatically).
            In general, passing categorical features is preferable to using one-hot encoding, for example.
        :param random_state: the random seed to use
        :param num_leaves: the maximum number of leaves in one tree (original lightgbm default is 31)
        :param modelArgs: see https://lightgbm.readthedocs.io/en/latest/Parameters.html
        """
        self.categoricalFeatureNames = categoricalFeatureNames
        super().__init__(lightgbm.sklearn.LGBMRegressor,
            sklearnInputTransformer=sklearnInputTransformer, sklearnOutputTransformer=sklearnOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers, targetTransformer=targetTransformer,
            random_state=random_state, num_leaves=num_leaves, **modelArgs)

    def _updateModelArgs(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        if self.categoricalFeatureNames is not None:
            cols = list(inputs.columns)
            colIndices = [cols.index(f) for f in self.categoricalFeatureNames]
            args = {"cat_column": colIndices}
            self.log.info(f"Updating model parameters with {args}")
            self.modelArgs.update(args)


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
