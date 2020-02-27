import copy
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn import compose

from ..vector_model import VectorRegressionModel, VectorClassificationModel

_log = logging.getLogger(__name__)


def createSkLearnModel(modelConstructor, modelArgs, outputTransformer=None):
    model = modelConstructor(**modelArgs)
    if outputTransformer is not None:
        model = compose.TransformedTargetRegressor(regressor=model, transformer=outputTransformer)
    return model


class AbstractSkLearnVectorRegressionModel(VectorRegressionModel, ABC):
    """
    Base class for models built upon scikit-learn's model implementations
    """
    _log = _log.getChild(__qualname__)

    def __init__(self, modelConstructor, **modelArgs):
        """
        :param modelConstructor: the sklearn model constructor
        :param modelArgs: arguments to be passed to the sklearn model constructor
        """
        super().__init__()
        self.sklearnInputTransformer = None
        self.sklearnOutputTransformer = None
        self.modelConstructor = modelConstructor
        self.modelArgs = modelArgs

    def withSkLearnInputTransformer(self, sklearnInputTransformer) -> __qualname__:
        """
        :param sklearnInputTransformer: an optional sklearn preprocessor for normalising/scaling inputs
        :return: self
        """
        self.sklearnInputTransformer = sklearnInputTransformer
        return self

    def withSkLearnOutputTransformer(self, sklearnOutputTransformer):
        """
        :param sklearnOutputTransformer: an optional sklearn preprocessor for normalising/scaling outputs
        :return: self
        """
        self.sklearnOutputTransformer = sklearnOutputTransformer
        return self

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
        inputs = self._transformInput(x)
        return self._predictSkLearn(inputs)

    @abstractmethod
    def _predictSkLearn(self, inputs: pd.DataFrame):
        pass


class AbstractSkLearnMultipleOneDimVectorRegressionModel(AbstractSkLearnVectorRegressionModel, ABC):
    """
    Base class for models which use several sklearn models of the same type with a single
    output dimension to create a multi-dimensional model (for the case where there is more than one output dimension)
    """
    def __init__(self, modelConstructor, **modelArgs):
        super().__init__(modelConstructor, **modelArgs)
        self.models = {}

    def __str__(self):
        if len(self.models) > 0:
            modelStr = str(next(iter(self.models.values())))
        else:
            modelStr = f"{self.modelConstructor.__name__}{self.modelArgs}"
        return f"{self.__class__.__name__}[{modelStr}]"

    def _fitSkLearn(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        for predictedVarName in outputs.columns:
            _log.info(f"Fitting model for output variable '{predictedVarName}'")
            model = createSkLearnModel(self.modelConstructor,
                    self.modelArgs,
                    outputTransformer=copy.deepcopy(self.sklearnOutputTransformer))
            model.fit(inputs, outputs[predictedVarName])
            self.models[predictedVarName] = model

    def _predictSkLearn(self, inputs: pd.DataFrame) -> pd.DataFrame:
        results = {}
        for varName in self.models:
            results[varName] = self.models[varName].predict(inputs)
        return pd.DataFrame(results)


class AbstractSkLearnMultiDimVectorRegressionModel(AbstractSkLearnVectorRegressionModel, ABC):
    """
    Base class for models which use a single sklearn model with multiple output dimensions to create the multi-dimensional model
    """
    def __init__(self, modelConstructor, **modelArgs):
        super().__init__(modelConstructor, **modelArgs)
        self.model = None

    def __str__(self):
        if self.model is not None:
            modelStr = str(self.model)
        else:
            modelStr = f"{self.modelConstructor.__name__}{self.modelArgs}"
        return f"{self.__class__.__name__}[{modelStr}]"

    def _fitSkLearn(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        if len(outputs.columns) > 1:
            _log.info(f"Fitting a single multi-dimensional model for all {len(outputs.columns)} output dimensions")
        self.model = createSkLearnModel(self.modelConstructor, self.modelArgs, outputTransformer=self.sklearnOutputTransformer)
        outputValues = outputs.values
        if outputValues.shape[1] == 1:  # for 1D output, shape must be (numSamples,) rather than (numSamples, 1)
            outputValues = np.ravel(outputValues)
        self.model.fit(inputs, outputValues)

    def _predictSkLearn(self, inputs: pd.DataFrame) -> pd.DataFrame:
        Y = self.model.predict(inputs)
        return pd.DataFrame(Y, columns=self.getModelOutputVariableNames())


class AbstractSkLearnVectorClassificationModel(VectorClassificationModel, ABC):
    def __init__(self, modelConstructor, **modelArgs):
        """
        :param modelConstructor: the sklearn model constructor
        :param modelArgs: arguments to be passed to the sklearn model constructor
        """
        super().__init__()
        self.modelConstructor = modelConstructor
        self.sklearnInputTransformer = None
        self.sklearnOutputTransformer = None
        self.modelArgs = modelArgs
        self.model = None

    def withSkLearnInputTransformer(self, sklearnInputTransformer) -> __qualname__:
        """
        :param sklearnInputTransformer: an optional sklearn preprocessor for normalising/scaling inputs
        :return: self
        """
        self.sklearnInputTransformer = sklearnInputTransformer
        return self

    def withSkLearnOutputTransformer(self, sklearnOutputTransformer):
        """
        :param sklearnOutputTransformer: an optional sklearn preprocessor for normalising/scaling outputs
        :return: self
        """
        self.sklearnOutputTransformer = sklearnOutputTransformer
        return self

    def __str__(self):
        if self.model is None:
            strModel = f"{self.modelConstructor.__name__}{self.modelArgs}"
        else:
            strModel = str(self.model)
        return f"{self.__class__.__name__}[{strModel}]"

    def _updateModelArgs(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        """
        Designed to be overridden in order to make input data-specific changes to modelArgs

        :param inputs: the training input data
        :param outputs: the training output data
        """
        pass

    def _fitClassifier(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        inputValues = self._transformInput(inputs, fit=True)
        self._updateModelArgs(inputs, outputs)
        self.model = createSkLearnModel(self.modelConstructor, self.modelArgs, self.sklearnOutputTransformer)
        _log.info(f"Fitting sklearn classifier of type {self.model.__class__.__name__}")
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

