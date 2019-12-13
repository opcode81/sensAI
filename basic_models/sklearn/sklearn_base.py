from typing import List, Sequence
import copy
import logging
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn import compose

from ..basic_models_base import VectorRegressionModel, VectorClassificationModel, DataFrameTransformer

log = logging.getLogger(__name__)


def createSkLearnModel(modelConstructor, modelArgs, outputTransformer=None):
    model = modelConstructor(**modelArgs)
    if outputTransformer is not None:
        model = compose.TransformedTargetRegressor(regressor=model, transformer=outputTransformer)
    return model


class AbstractSkLearnVectorRegressionModel(VectorRegressionModel, ABC):
    """
    Base class for models built upon scikit-learn's model implementations
    """

    log = log.getChild(__qualname__)

    def __init__(self, modelConstructor, modelInputTransformer=None, modelOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (), outputTransformers: Sequence[DataFrameTransformer] = (),
            trainingOutputTransformers: Sequence[DataFrameTransformer] = (), **modelArgs):
        """
        :param modelConstructor: the sklearn model constructor
        :param modelInputTransformer: an optional sklearn preprocessor for normalising/scaling inputs
        :param modelInputTransformer: an optional sklearn preprocessor for normalising/scaling outputs
        :param modelArgs: arguments to be passed to the sklearn model constructor
        :param inputTransformers: list of DataFrameTransformers for the transformation of inputs
        :param outputTransformers: list of DataFrameTransformers for the transformation of outputs
        :param trainingOutputTransformers: list of DataFrameTransformers for the transformation of training outputs prior to training
        """
        # TODO Consider replacing modelInputTransformers with new DataFrameTransformer interface, but it might not we elegantly possible, because sklearn's mechanisms apply to arrays not DataFrames (no column names!)
        super().__init__(inputTransformers=inputTransformers, outputTransformers=outputTransformers,
            trainingOutputTransformers=trainingOutputTransformers)
        self.modelInputTransformer = modelInputTransformer
        self.modelOutputTransformer = modelOutputTransformer
        self.modelConstructor = modelConstructor
        self.modelArgs = modelArgs

    def _transformInput(self, inputs: pd.DataFrame, fit=False) -> np.ndarray:
        inputValues = inputs.values
        if self.modelInputTransformer is not None:
            if fit:
                inputValues = self.modelInputTransformer.fit_transform(inputValues)
            else:
                inputValues = self.modelInputTransformer.transform(inputValues)
        return inputValues

    def _fit(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        inputValues = self._transformInput(inputs, fit=True)
        self._fitSkLearn(inputValues, outputs)

    @abstractmethod
    def _fitSkLearn(self, inputValues: np.ndarray, outputs: pd.DataFrame):
        pass

    def _predict(self, x: pd.DataFrame):
        inputValues = self._transformInput(x)
        return self._predictSkLearn(inputValues)

    @abstractmethod
    def _predictSkLearn(self, inputValues: np.ndarray):
        pass


class AbstractSkLearnMultipleOneDimVectorRegressionModel(AbstractSkLearnVectorRegressionModel, ABC):
    """
    Base class for models which use several sklearn models of the same type with a single
    output dimension to create a multi-dimensional model (for the case where there is more than one output dimension)
    """

    def __init__(self, modelConstructor, modelInputTransformer=None, modelOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (), outputTransformers: Sequence[DataFrameTransformer] = (),
            trainingOutputTransformers: Sequence[DataFrameTransformer] = (), **modelArgs):
        super().__init__(modelConstructor, modelInputTransformer=modelInputTransformer, modelOutputTransformer=modelOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers, trainingOutputTransformers=trainingOutputTransformers,
            **modelArgs)
        self.models = {}

    def __str__(self):
        modelStr = str(next(iter(self.models.values()))) if len(self.models) > 0 else "None"
        return f"{self.__class__.__name__}[{modelStr}]"

    def _fitSkLearn(self, inputValues: np.ndarray, outputs: pd.DataFrame):
        for predictedVarName in outputs.columns:
            log.info(f"Fitting model for output variable '{predictedVarName}'")
            model = createSkLearnModel(self.modelConstructor,
                    self.modelArgs,
                    outputTransformer=copy.deepcopy(self.modelOutputTransformer))
            model.fit(inputValues, outputs[predictedVarName].values)
            self.models[predictedVarName] = model

    def _predictSkLearn(self, inputValues) -> pd.DataFrame:
        results = {}
        for varName in self.models:
            results[varName] = self.models[varName].predict(inputValues)
        return pd.DataFrame(results)


class AbstractSkLearnMultiDimVectorRegressionModel(AbstractSkLearnVectorRegressionModel, ABC):
    """
    Base class for models which use a single sklearn model with multiple output dimensions to create the multi-dimensional model
    """

    def __init__(self, modelConstructor, modelInputTransformer=None, modelOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (), outputTransformers: Sequence[DataFrameTransformer] = (),
            trainingOutputTransformers: Sequence[DataFrameTransformer] = (), **modelArgs):
        super().__init__(modelConstructor, modelInputTransformer=modelInputTransformer, modelOutputTransformer=modelOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers, trainingOutputTransformers=trainingOutputTransformers,
            **modelArgs)

    def __str__(self):
        return f"{self.__class__.__name__}[{self.model}]"

    def _fitSkLearn(self, inputValues: np.ndarray, outputs: pd.DataFrame):
        log.info("Fitting multi-dimensional model")
        self.model = createSkLearnModel(self.modelConstructor,
                self.modelArgs,
                outputTransformer=self.modelOutputTransformer)
        self.model.fit(inputValues, outputs.values)

    def _predictSkLearn(self, inputValues) -> pd.DataFrame:
        Y = self.model.predict(inputValues)
        return pd.DataFrame(Y, columns=self.getModelOutputVariableNames())


class AbstractSkLearnVectorClassificationModel(VectorClassificationModel, ABC):
    def __init__(self, modelConstructor, sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (), outputTransformers: Sequence[DataFrameTransformer] = (),
            trainingOutputTransformers: Sequence[DataFrameTransformer] = (), **modelArgs):
        """
        Abstract base model with additional prediction of class probabilities
        :param modelConstructor: the sklearn model constructor
        :param inputTransformers: a list of transformers to apply to input DataFrames
        :param modelArgs: arguments to be passed to the sklearn model constructor
        """
        super().__init__(inputTransformers=inputTransformers, outputTransformers=outputTransformers,
            trainingOutputTransformers=trainingOutputTransformers)
        self.modelConstructor = modelConstructor
        self.sklearnInputTransformer = sklearnInputTransformer
        self.sklearnOutputTransformer = sklearnOutputTransformer
        self.modelArgs = modelArgs
        self.model = createSkLearnModel(self.modelConstructor, self.modelArgs, outputTransformer=sklearnOutputTransformer)

    def __str__(self):
        return f"{self.__class__.__name__}[{str(self.model)}]"

    def _fitClassifier(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        inputValues = self._transformInput(inputs, fit=True)
        self.model = createSkLearnModel(self.modelConstructor, self.modelArgs)
        log.info(f"Fitting sklearn classifier of type {self.model.__class__.__name__}")
        self.model.fit(inputValues, np.ravel(outputs.values))

    def _transformInput(self, inputs: pd.DataFrame, fit=False) -> np.ndarray:
        inputValues = inputs.values
        if self.sklearnInputTransformer is not None:
            if fit:
                inputValues = self.sklearnInputTransformer.fit_transform(inputValues)
            else:
                inputValues = self.sklearnInputTransformer.transform(inputValues)
        return inputValues

    def _predict(self, x: pd.DataFrame):
        inputValues = self._transformInput(x)
        Y = self.model.predict(inputValues)
        return pd.DataFrame(Y, columns=self._predictedVariableNames)

    def _predictClassProbabilities(self, x: pd.DataFrame):
        inputValues = self._transformInput(x)
        Y = self.model.predict_proba(inputValues)
        return pd.DataFrame(Y, columns=self._labels)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)

