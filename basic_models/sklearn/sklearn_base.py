import copy
import logging
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn import compose

from ..basic_models_base import VectorRegressionModel, VectorClassificationModel
from ..data_transformation import DataFrameTransformer, InvertibleDataFrameTransformer

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

    def __init__(self, modelConstructor, sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (), outputTransformers: Sequence[DataFrameTransformer] = (),
            targetTransformer: InvertibleDataFrameTransformer = None, **modelArgs):
        """
        :param modelConstructor: the sklearn model constructor
        :param sklearnInputTransformer: an optional sklearn preprocessor for normalising/scaling inputs
        :param sklearnOutputTransformer: an optional sklearn preprocessor for normalising/scaling outputs
        :param inputTransformers: list of DataFrameTransformers for the transformation of inputs
        :param outputTransformers: list of DataFrameTransformers for the transformation of outputs
        :param targetTransformer: a transformer which transforms the targets (training data outputs) prior to learning the model, such
            that the model learns to predict the transformed outputs. When predicting, the inverse transformer is applied after applying
            the model, i.e. the transformation is completely transparent when applying the model.
        :param modelArgs: arguments to be passed to the sklearn model constructor
        """
        super().__init__(inputTransformers=inputTransformers, outputTransformers=outputTransformers,
            targetTransformer=targetTransformer)
        self.sklearnInputTransformer = sklearnInputTransformer
        self.sklearnOutputTransformer = sklearnOutputTransformer
        self.modelConstructor = modelConstructor
        self.modelArgs = modelArgs

    def _transformInput(self, inputs: pd.DataFrame, fit=False) -> pd.DataFrame:
        if self.sklearnInputTransformer is None:
            return inputs
        else:
            inputValues = inputs.values
            shapeBefore = inputValues.shape
            if fit:
                inputValues = self.sklearnInputTransformer.fit_transform(inputValues)
            else:
                inputValues = self.sklearnInputTransformer.transform(inputValues)
            if inputValues.shape != shapeBefore:
                raise Exception("sklearnInputTransformer changed the shape of the input, which is unsupported. Consider using an a DFTSkLearnTransformer in inputTransformers instead.")
            return pd.DataFrame(inputValues, index=inputs.index, columns=inputs.columns)

    def _updateModelArgs(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        """
        Designed to be overridden in order to make input data-specific changes to modelArgs

        :param inputs: the training input data
        :param outputs: the training output data
        """
        pass

    def _fit(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        inputs = self._transformInput(inputs, fit=True)
        self._updateModelArgs(inputs, outputs)
        self._fitSkLearn(inputs, outputs)

    @abstractmethod
    def _fitSkLearn(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        pass

    def _predict(self, x: pd.DataFrame):
        inputValues = self._transformInput(x).values
        return self._predictSkLearn(inputValues)

    @abstractmethod
    def _predictSkLearn(self, inputValues: np.ndarray):
        pass


class AbstractSkLearnMultipleOneDimVectorRegressionModel(AbstractSkLearnVectorRegressionModel, ABC):
    """
    Base class for models which use several sklearn models of the same type with a single
    output dimension to create a multi-dimensional model (for the case where there is more than one output dimension)
    """
    def __init__(self, modelConstructor, sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (), outputTransformers: Sequence[DataFrameTransformer] = (),
            targetTransformer: InvertibleDataFrameTransformer = None,
            **modelArgs):
        super().__init__(modelConstructor,
            sklearnInputTransformer=sklearnInputTransformer, sklearnOutputTransformer=sklearnOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers, targetTransformer=targetTransformer,
            **modelArgs)
        self.models = {}

    def __str__(self):
        if len(self.models) > 0:
            modelStr = str(next(iter(self.models.values())))
        else:
            modelStr = f"{self.modelConstructor.__name__}{self.modelArgs}"
        return f"{self.__class__.__name__}[{modelStr}]"

    def _fitSkLearn(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        for predictedVarName in outputs.columns:
            log.info(f"Fitting model for output variable '{predictedVarName}'")
            model = createSkLearnModel(self.modelConstructor,
                    self.modelArgs,
                    outputTransformer=copy.deepcopy(self.sklearnOutputTransformer))
            model.fit(inputs, outputs[predictedVarName])
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
    def __init__(self, modelConstructor,
            sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (), outputTransformers: Sequence[DataFrameTransformer] = (),
            targetTransformer: InvertibleDataFrameTransformer = None,
            **modelArgs):
        super().__init__(modelConstructor,
            sklearnInputTransformer=sklearnInputTransformer, sklearnOutputTransformer=sklearnOutputTransformer,
            inputTransformers=inputTransformers, outputTransformers=outputTransformers, targetTransformer=targetTransformer,
            **modelArgs)
        self.model = None

    def __str__(self):
        if self.model is not None:
            modelStr = str(self.model)
        else:
            modelStr = f"{self.modelConstructor.__name__}{self.modelArgs}"
        return f"{self.__class__.__name__}[{modelStr}]"

    def _fitSkLearn(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        if len(outputs.columns) > 1:
            log.info(f"Fitting a single multi-dimensional model for all {len(outputs.columns)} output dimensions")
        self.model = createSkLearnModel(self.modelConstructor, self.modelArgs, outputTransformer=self.sklearnOutputTransformer)
        outputValues = outputs.values
        if outputValues.shape[1] == 1:  # for 1D output, shape must be (numSamples,) rather than (numSamples, 1)
            outputValues = np.ravel(outputValues)
        self.model.fit(inputs, outputValues)

    def _predictSkLearn(self, inputValues) -> pd.DataFrame:
        Y = self.model.predict(inputValues)
        return pd.DataFrame(Y, columns=self.getModelOutputVariableNames())


class AbstractSkLearnVectorClassificationModel(VectorClassificationModel, ABC):
    def __init__(self, modelConstructor,
            sklearnInputTransformer=None, sklearnOutputTransformer=None,
            inputTransformers: Sequence[DataFrameTransformer] = (), outputTransformers: Sequence[DataFrameTransformer] = (),
            **modelArgs):
        """
        :param modelConstructor: the sklearn model constructor
        :param sklearnInputTransformer: an optional sklearn preprocessor for normalising/scaling inputs
        :param sklearnOutputTransformer: an optional sklearn preprocessor for normalising/scaling outputs
        :param inputTransformers: list of DataFrameTransformers for the transformation of inputs
        :param outputTransformers: list of DataFrameTransformers for the transformation of outputs
        :param modelArgs: arguments to be passed to the sklearn model constructor
        """
        super().__init__(inputTransformers=inputTransformers, outputTransformers=outputTransformers)
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

