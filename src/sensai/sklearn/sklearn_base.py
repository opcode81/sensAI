import copy
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn import compose

from ..feature_importance import FeatureImportanceProvider
from ..util.pickle import setstate
from ..util.string import dictString
from ..vector_model import VectorRegressionModel, VectorClassificationModel

log = logging.getLogger(__name__)


def createSkLearnModel(modelConstructor, modelArgs, outputTransformer=None):
    model = modelConstructor(**modelArgs)
    if outputTransformer is not None:
        model = compose.TransformedTargetRegressor(regressor=model, transformer=outputTransformer)
    return model


def strSkLearnModel(model):
    """
    Creates a cleaned string representation of the model with line breaks and indentations removed

    :param model: the sklearn model for which to generate the cleaned string representation
    :return: the string representation
    """
    return re.sub(r",\s*", ", ", str(model))


def _applySkLearnInputTransformer(inputs: pd.DataFrame, sklearnInputTransformer: Optional, fit: bool) -> pd.DataFrame:
    if sklearnInputTransformer is None:
        return inputs
    else:
        inputValues = inputs.values
        shapeBefore = inputValues.shape
        if fit:
            inputValues = sklearnInputTransformer.fit_transform(inputValues)
        else:
            inputValues = sklearnInputTransformer.transform(inputValues)
        if inputValues.shape != shapeBefore:
            raise Exception("sklearnInputTransformer changed the shape of the input, which is unsupported. Consider using an a DFTSkLearnTransformer in inputTransformers instead.")
        return pd.DataFrame(inputValues, index=inputs.index, columns=inputs.columns)


class AbstractSkLearnVectorRegressionModel(VectorRegressionModel, ABC):
    """
    Base class for models built upon scikit-learn's model implementations
    """
    log = log.getChild(__qualname__)

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
        self.fitArgs = {}

    def _toStringExcludes(self) -> List[str]:
        return super()._toStringExcludes() + ["sklearnInputTransformer", "sklearnOutputTransformer", "modelConstructor", "modelArgs"]

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
        return _applySkLearnInputTransformer(inputs, self.sklearnInputTransformer, fit)

    def _updateModelArgs(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        """
        Designed to be overridden in order to make input data-specific changes to modelArgs

        :param inputs: the training input data
        :param outputs: the training output data
        """
        pass

    def _updateFitArgs(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        """
        Designed to be overridden in order to make input data-specific changes to fitArgs (arguments to be passed to the
        underlying model's fit method)

        :param inputs: the training input data
        :param outputs: the training output data
        """
        pass

    def _fit(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        inputs = self._transformInput(inputs, fit=True)
        self._updateModelArgs(inputs, outputs)
        self._updateFitArgs(inputs, outputs)
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

    def _toStringExcludes(self) -> List[str]:
        return super()._toStringExcludes() + ["models"]

    def _toStringAdditionalEntries(self) -> Dict[str, Any]:
        d = super()._toStringAdditionalEntries()
        if len(self.models) > 0:
            d["model[0]"] = strSkLearnModel(next(iter(self.models.values())))
        else:
            d["modelConstructor"] = f"{self.modelConstructor.__name__}({dictString(self.modelArgs)})"
        return d

    def _fitSkLearn(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        for predictedVarName in outputs.columns:
            log.info(f"Fitting model for output variable '{predictedVarName}'")
            model = createSkLearnModel(self.modelConstructor,
                    self.modelArgs,
                    outputTransformer=copy.deepcopy(self.sklearnOutputTransformer))
            model.fit(inputs, outputs[predictedVarName], **self.fitArgs)
            self.models[predictedVarName] = model

    def _predictSkLearn(self, inputs: pd.DataFrame) -> pd.DataFrame:
        results = {}
        for varName in self.models:
            results[varName] = self.models[varName].predict(inputs)
        return pd.DataFrame(results)

    def getSkLearnModel(self, predictedVarName=None):
        if predictedVarName is None:
            if len(self.models) > 1:
                raise ValueError(f"Must provide predicted variable name (one of {self.models.keys()})")
            return next(iter(self.models.values()))
        return self.models[predictedVarName]


class AbstractSkLearnMultiDimVectorRegressionModel(AbstractSkLearnVectorRegressionModel, ABC):
    """
    Base class for models which use a single sklearn model with multiple output dimensions to create the multi-dimensional model
    """
    def __init__(self, modelConstructor, **modelArgs):
        super().__init__(modelConstructor, **modelArgs)
        self.model = None

    def _toStringExcludes(self) -> List[str]:
        return super()._toStringExcludes() + ["model"]

    def _toStringAdditionalEntries(self) -> Dict[str, Any]:
        d = super()._toStringAdditionalEntries()
        if self.model is not None:
            d["model"] = strSkLearnModel(self.model)
        else:
            d["modelConstructor"] = f"{self.modelConstructor.__name__}({dictString(self.modelArgs)})"
        return d

    def _fitSkLearn(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        if len(outputs.columns) > 1:
            log.info(f"Fitting a single multi-dimensional model for all {len(outputs.columns)} output dimensions")
        self.model = createSkLearnModel(self.modelConstructor, self.modelArgs, outputTransformer=self.sklearnOutputTransformer)
        outputValues = outputs.values
        if outputValues.shape[1] == 1:  # for 1D output, shape must be (numSamples,) rather than (numSamples, 1)
            outputValues = np.ravel(outputValues)
        self.model.fit(inputs, outputValues, **self.fitArgs)

    def _predictSkLearn(self, inputs: pd.DataFrame) -> pd.DataFrame:
        Y = self.model.predict(inputs)
        return pd.DataFrame(Y, columns=self.getModelOutputVariableNames())


class AbstractSkLearnVectorClassificationModel(VectorClassificationModel, ABC):
    def __init__(self, modelConstructor, useBalancedClassWeights=False, **modelArgs):
        """
        :param modelConstructor: the sklearn model constructor
        :param modelArgs: arguments to be passed to the sklearn model constructor
        :param useBalancedClassWeights: whether to compute class weights from the training data and apply the corresponding weight to
            each data point such that the sum of weights for all classes is equal. This is achieved by applying a weight proportional
            to the reciprocal frequency of the class in the (training) data. We scale weights such that the smallest weight (of the
            largest class) is 1, ensuring that weight counts still reasonably correspond to data point counts.
            Note that weighted data points may not be supported for all types of models.
        """
        super().__init__()
        self.modelConstructor = modelConstructor
        self.sklearnInputTransformer = None
        self.sklearnOutputTransformer = None
        self.modelArgs = modelArgs
        self.fitArgs = {}
        self.useBalancedClassWeights = useBalancedClassWeights
        self.model = None

    def __setstate__(self, state):
        setstate(AbstractSkLearnVectorClassificationModel, self, state, newDefaultProperties={"useComputedClassWeights": False},
            renamedProperties={"useComputedClassWeights": "useBalancedClassWeights"})

    def _toStringExcludes(self) -> List[str]:
        return super()._toStringExcludes() + ["modelConstructor", "sklearnInputTransformer", "sklearnOutputTransformer",
            "modelArgs", "model"]

    def _toStringAdditionalEntries(self) -> Dict[str, Any]:
        d = super()._toStringAdditionalEntries()
        if self.model is None:
            d["modelConstructor"] = f"{self.modelConstructor.__name__}({dictString(self.modelArgs)})"
        else:
            d["model"] = strSkLearnModel(self.model)
        return d

    def withSkLearnInputTransformer(self, sklearnInputTransformer) -> __qualname__:
        """
        :param sklearnInputTransformer: an optional sklearn preprocessor for transforming inputs
        :return: self
        """
        self.sklearnInputTransformer = sklearnInputTransformer
        return self

    def withSkLearnOutputTransformer(self, sklearnOutputTransformer):
        """
        :param sklearnOutputTransformer: an optional sklearn preprocessor for transforming the target labels
        :return: self
        """
        raise Exception("This is currently not supported for classifiers")  # TODO add support
        self.sklearnOutputTransformer = sklearnOutputTransformer
        return self

    def _updateModelArgs(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        """
        Designed to be overridden in order to make input data-specific changes to modelArgs

        :param inputs: the training input data
        :param outputs: the training output data
        """
        pass

    def _updateFitArgs(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        """
        Designed to be overridden in order to make input data-specific changes to fitArgs (arguments to be passed to the
        underlying model's fit method)

        :param inputs: the training input data
        :param outputs: the training output data
        """
        pass

    def _fitClassifier(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        inputs = self._transformInput(inputs, fit=True)
        self._updateModelArgs(inputs, outputs)
        self._updateFitArgs(inputs, outputs)
        self.model = createSkLearnModel(self.modelConstructor, self.modelArgs, self.sklearnOutputTransformer)
        log.info(f"Fitting sklearn classifier of type {self.model.__class__.__name__}")
        kwargs = dict(self.fitArgs)
        if self.useBalancedClassWeights:
            class2weight = self._computeClassWeights(outputs)
            classes = outputs.iloc[:, 0]
            weights = np.array([class2weight[cls] for cls in classes])
            weights = weights / np.min(weights)
            kwargs["sample_weight"] = weights
        outputValues = np.ravel(outputs.values)
        self.model.fit(inputs, outputValues, **kwargs)

    def _transformInput(self, inputs: pd.DataFrame, fit=False) -> pd.DataFrame:
        return _applySkLearnInputTransformer(inputs, self.sklearnInputTransformer, fit)

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

    def _computeClassWeights(self, outputs: pd.DataFrame):
        """
        :param outputs: the output data frame containing the class labels as the first column
        :return: the dictionary of class weights mapping class to weight value
        """
        classes: pd.Series = outputs.iloc[:,0]
        counts = classes.value_counts()
        rfreqs = counts / counts.sum()
        weights: pd.Series = 1.0 / rfreqs
        return weights.to_dict()


def _getModelFeatureImportanceVector(model):
    candAttributes = ("feature_importances_", "coef_")
    for attr in candAttributes:
        if hasattr(model, attr):
            importanceValues = getattr(model, attr)
            if attr == "coef_":
                importanceValues = np.abs(importanceValues)  # for coefficients in linear models, use the absolute values
            return importanceValues
    raise ValueError(f"Model {model} has none of the attributes {candAttributes}")


class FeatureImportanceProviderSkLearnRegressionMultipleOneDim(FeatureImportanceProvider):
    def getFeatureImportanceDict(self) -> Dict[str, Dict[str, int]]:
        self: AbstractSkLearnMultipleOneDimVectorRegressionModel
        return {targetFeature: dict(zip(self._modelInputVariableNames, _getModelFeatureImportanceVector(model))) for targetFeature, model in self.models.items()}


class FeatureImportanceProviderSkLearnRegressionMultiDim(FeatureImportanceProvider):
    def getFeatureImportanceDict(self) -> Dict[str, float]:
        self: AbstractSkLearnMultiDimVectorRegressionModel
        return dict(zip(self._modelInputVariableNames, _getModelFeatureImportanceVector(self.model)))


class FeatureImportanceProviderSkLearnClassification(FeatureImportanceProvider):
    def getFeatureImportanceDict(self) -> Dict[str, float]:
        self: AbstractSkLearnVectorClassificationModel
        return dict(zip(self._modelInputVariableNames, _getModelFeatureImportanceVector(self.model)))
