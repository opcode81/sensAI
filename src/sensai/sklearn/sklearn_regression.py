import logging
from typing import Union, Optional, Dict

import sklearn.ensemble
import sklearn.linear_model
import sklearn.neighbors
import sklearn.neural_network
import sklearn.svm

from .sklearn_base import AbstractSkLearnMultipleOneDimVectorRegressionModel, AbstractSkLearnMultiDimVectorRegressionModel


log = logging.getLogger(__name__)


class SkLearnRandomForestVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    def __init__(self, n_estimators=100, min_samples_leaf=10, random_state=42, **modelArgs):
        super().__init__(sklearn.ensemble.RandomForestRegressor,
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=random_state, **modelArgs)

    def getFeatureImportances(self) -> Dict[str, Dict[str, float]]:
        return {targetFeature: dict(zip(self._modelInputVariableNames, model.feature_importances_)) for targetFeature, model in self.models.items()}


class SkLearnLinearRegressionVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self, **modelArgs):
        super().__init__(sklearn.linear_model.LinearRegression, **modelArgs)

    def getFeatureImportances(self) -> Dict[str, float]:
        return dict(zip(self._modelInputVariableNames, self.model.feature_importances_))


class SkLearnMultiLayerPerceptronVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self,
            hidden_layer_sizes=(100,), activation: str = "relu",
            solver: str = "adam", batch_size: Union[int, str] = "auto", random_state: Optional[int] = 42,
            max_iter: int = 200, early_stopping: bool = False, n_iter_no_change: int = 10, **modelArgs):
        """
        :param hidden_layer_sizes: the sequence of hidden layer sizes
        :param activation: {"identity", "logistic", "tanh", "relu"} the activation function to use for hidden layers (the one used for the output layer is always 'identity')
        :param solver: {"adam", "lbfgs", "sgd"} the name of the solver to apply
        :param batch_size: the batch size or "auto" for min(200, data set size)
        :param random_state: the random seed for reproducability; use None if it shall not be specifically defined
        :param max_iter: the number of iterations (gradient steps for L-BFGS, epochs for other solvers)
        :param early_stopping: whether to use early stopping (stop training after n_iter_no_change epochs without improvement)
        :param n_iter_no_change: the number of iterations after which to stop early (if early_stopping is enabled)
        :param modelArgs: additional arguments to pass on to MLPRegressor, see https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
        """
        super().__init__(sklearn.neural_network.MLPRegressor,
            random_state=random_state, hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, batch_size=batch_size, max_iter=max_iter,
            early_stopping=early_stopping, n_iter_no_change=n_iter_no_change, **modelArgs)


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
