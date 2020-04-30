import logging

import sklearn.ensemble
import sklearn.linear_model
import sklearn.neighbors
import sklearn.neural_network
import sklearn.svm

from .sklearn_base import AbstractSkLearnMultipleOneDimVectorRegressionModel, AbstractSkLearnMultiDimVectorRegressionModel


_log = logging.getLogger(__name__)


class SkLearnRandomForestVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    def __init__(self, n_estimators=100, min_samples_leaf=10, random_state=42, **modelArgs):
        super().__init__(sklearn.ensemble.RandomForestRegressor,
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=random_state, **modelArgs)


class SkLearnLinearRegressionVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self, **modelArgs):
        super().__init__(sklearn.linear_model.LinearRegression, **modelArgs)


class SkLearnMultiLayerPerceptronVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self,
            random_state=42, hidden_layer_sizes=(100,), activation="relu", solver="adam",
            max_iter=200, **modelArgs):
        """
        :param random_state: the random seed
        :param hidden_layer_sizes: the sequence of hidden layer sizes
        :param activation: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’} the activation function to use for hidden layers
            (the one used for the output layer is always 'identity')
        :param solver: {‘lbfgs’, ‘sgd’, ‘adam’} the name of the solver to apply
        :param max_iter: the number of iterations (gradient steps for L-BFGS, epochs for other solvers)
        :param modelArgs: additional arguments to pass on to MLPRegressor, see https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
        """
        super().__init__(sklearn.neural_network.MLPRegressor,
            random_state=random_state, hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter,
            **modelArgs)


class SkLearnSVRVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self, **modelArgs):
        super().__init__(sklearn.svm.SVR, **modelArgs)


class SkLearnLinearSVRVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self, **modelArgs):
        super().__init__(sklearn.svm.LinearSVR, **modelArgs)


class SkLearnGradientBoostingVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    def __init__(self, random_state=42, **modelArgs):
        super().__init__(sklearn.ensemble.GradientBoostingRegressor, random_state=random_state, **modelArgs)


class SkLearnKNeighborsVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self, **modelArgs):
        super().__init__(sklearn.neighbors.KNeighborsRegressor, **modelArgs)


class SkLearnExtraTreesVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    def __init__(self, n_estimators=100, min_samples_leaf=10, random_state=42, **modelArgs):
        super().__init__(sklearn.ensemble.ExtraTreesRegressor,
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=random_state, **modelArgs)
