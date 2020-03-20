import logging
from abc import ABC, abstractmethod
from typing import Sequence, List, Any, Optional, Union, TypeVar

import numpy as np
import pandas as pd
import scipy.stats

from .data_transformation import DataFrameTransformer, DataFrameTransformerChain, InvertibleDataFrameTransformer
from .featuregen import FeatureGenerator, FeatureCollector

_log = logging.getLogger(__name__)


T = TypeVar('T')


class InputOutputData:
    def __init__(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        if len(inputs) != len(outputs):
            raise ValueError("Lengths do not match")
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    @property
    def inputDim(self):
        return self.inputs.shape[1]

    @property
    def outputDim(self):
        return self.outputs.shape[1]

    def filterIndices(self, indices: Sequence[int]) -> 'InputOutputData':
        inputs = self.inputs.iloc[indices]
        outputs = self.outputs.iloc[indices]
        return InputOutputData(inputs, outputs)

    def computeInputOutputCorrelation(self):
        correlations = {}
        for outputCol in self.outputs.columns:
            correlations[outputCol] = {}
            outputSeries = self.outputs[outputCol]
            for inputCol in self.inputs.columns:
                inputSeries = self.inputs[inputCol]
                pcc, pvalue = scipy.stats.pearsonr(inputSeries, outputSeries)
                correlations[outputCol][inputCol] = pcc
        return correlations


class PredictorModel(ABC):
    """
    Base class for models that map vectors to predictions
    """
    @abstractmethod
    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def getPredictedVariableNames(self):
        pass

    @abstractmethod
    def isRegressionModel(self) -> bool:
        pass

    @abstractmethod
    def getName(self) -> str:
        pass


class VectorModel(PredictorModel, ABC):
    """
    Base class for models that map vectors to vectors
    """
    def __init__(self, checkInputColumns=True):
        """

        :param checkInputColumns: Whether to check if the input column list (after feature generation) during inference coincides
            with the input column list during fit. This should be disabled if feature generation is not performed by the model itself,
            e.g. in ensemble models.
        """
        self._featureGenerator: Optional["FeatureGenerator"] = None
        self._inputTransformerChain = DataFrameTransformerChain(())
        self._outputTransformerChain = DataFrameTransformerChain(())
        self._predictedVariableNames = None
        self._modelInputVariableNames = None
        self._modelOutputVariableNames = None
        self._targetTransformer = None
        self._name = None
        self._isFitted = False
        self.checkInputColumns = checkInputColumns

    @staticmethod
    def _flattened(l: Sequence[Union[T, List[T]]]) -> List[T]:
        result = []
        for x in l:
            if type(x) == list:
                result.extend(x)
            else:
                result.append(x)
        return result

    def withInputTransformers(self, *inputTransformers: Union[DataFrameTransformer, List[DataFrameTransformer]]) -> __qualname__:
        """
        Makes the model use the given input transformers.

        :param inputTransformers: DataFrameTransformers for the transformation of inputs
        :return: self
        """
        self._inputTransformerChain = DataFrameTransformerChain(self._flattened(inputTransformers))
        return self

    def withOutputTransformers(self, *outputTransformers: Union[DataFrameTransformer, List[DataFrameTransformer]]) -> __qualname__:
        """
        Makes the model use the given output transformers.

        :param outputTransformers: DataFrameTransformers for the transformation of outputs (after the model has been applied)
        :return: self
        """
        self._outputTransformerChain = DataFrameTransformerChain(self._flattened(outputTransformers))
        return self

    def withTargetTransformer(self, targetTransformer: InvertibleDataFrameTransformer) -> __qualname__:
        """
        Makes the model use the given target transformers.

        :param targetTransformer: a transformer which transforms the targets (training data outputs) prior to learning the model, such
            that the model learns to predict the transformed outputs. When predicting, the inverse transformer is applied after applying
            the model, i.e. the transformation is completely transparent when applying the model.
        :return: self
        """
        self._targetTransformer = targetTransformer
        return self

    def withFeatureGenerator(self, featureGenerator: Optional[FeatureGenerator]) -> __qualname__:
        """
        Makes the model use the given feature generator, which shall be used to compute the actual inputs of the model from the data
        frame that is given. Feature computation takes place before input transformation.

        :param featureGenerator: the feature generator to use for input computation
        :return: self
        """
        self._featureGenerator = featureGenerator
        return self

    def withFeatureCollector(self, featureCollector: FeatureCollector) -> __qualname__:
        """
        Makes the model use the given feature collector's multi-feature generator in order compute the actual inputs of the model from
        the data frame that is given.
        Feature computation takes place before input transformation.

        :param featureCollector: the feature collector whose feature generator shall be used for input computation
        :return: self
        """
        self._featureGenerator = featureCollector.getMultiFeatureGenerator()
        return self

    def withName(self, name: str):
        """
        Sets the model's name.

        :param name: the name
        :return: self
        """
        self.setName(name)
        return self

    @abstractmethod
    def isRegressionModel(self) -> bool:
        pass

    def isFitted(self):
        return self._isFitted

    def _computeInputs(self, x: pd.DataFrame, y=None) -> pd.DataFrame:
        fit = y is not None
        if self._featureGenerator is not None:
            if fit:
                x = self._featureGenerator.fitGenerate(x, y, self)
            else:
                x = self._featureGenerator.generate(x, self)
        x = self._inputTransformerChain.apply(x, fit=fit)
        if not fit:
            if not self.isFitted():
                raise Exception(f"Model has not been fitted")
            if self.checkInputColumns and list(x.columns) != self._modelInputVariableNames:
                raise Exception(f"Inadmissible input data frame: expected columns {self._modelInputVariableNames}, got {list(x.columns)}")
        return x

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Performs a prediction for the given input data frame

        :param x: the input data
        :return: a DataFrame with the same index as the input
        """
        x = self._computeInputs(x)
        y = self._predict(x)
        y.index = x.index
        y = self._outputTransformerChain.apply(y)
        if self._targetTransformer is not None:
            y = self._targetTransformer.applyInverse(y)
        return y

    @abstractmethod
    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        """
        Fits the model using the given data

        :param X: a data frame containing input data
        :param Y: a data frame containing output data
        """
        _log.info(f"Training {self.__class__.__name__}")
        self._predictedVariableNames = list(Y.columns)
        X = self._computeInputs(X, y=Y)
        if self._targetTransformer is not None:
            self._targetTransformer.fit(Y)
            Y = self._targetTransformer.apply(Y)
        self._modelInputVariableNames = list(X.columns)
        self._modelOutputVariableNames = list(Y.columns)
        _log.info(f"Training with outputs[{len(self._modelOutputVariableNames)}]={self._modelOutputVariableNames}, inputs[{len(self._modelInputVariableNames)}]=[{', '.join([n + '/' + X[n].dtype.name for n in self._modelInputVariableNames])}]")
        self._fit(X, Y)
        self._isFitted = True

    @abstractmethod
    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        pass

    def getPredictedVariableNames(self):
        return self._predictedVariableNames

    def getModelOutputVariableNames(self):
        """
        Gets the list of variable names predicted by the underlying model.
        For the case where the final output is transformed by an output transformer which changes column names,
        the names of the variables prior to the transformation will be returned, i.e. this method
        always returns the variable names that are actually predicted by the model.
        For the variable names that are ultimately output by the model (including output transformations),
        use getPredictedVariabaleNames.
        """
        return self._modelOutputVariableNames

    def getInputTransformer(self, cls):
        for it in self._inputTransformerChain.dataFrameTransformers:
            if isinstance(it, cls):
                return it
        return None

    def setName(self, name):
        self._name = name

    def getName(self):
        if self._name is None:
            return "unnamed-%x" % id(self)
        return self._name

    def setFeatureGenerator(self, featureGenerator: Optional[FeatureGenerator]):
        self.withFeatureGenerator(featureGenerator)

    def getFeatureGenerator(self) -> Optional[FeatureGenerator]:
        return self._featureGenerator


class VectorRegressionModel(VectorModel, ABC):
    def __init__(self):
        """
        """
        super().__init__()

    def isRegressionModel(self) -> bool:
        return True


class VectorClassificationModel(VectorModel, ABC):

    def __init__(self):
        """
        """
        self._labels = None
        super().__init__()

    def isRegressionModel(self) -> bool:
        return False

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        if len(Y.columns) != 1:
            raise ValueError("Classification requires exactly one output column with class labels")
        self._labels = sorted([label for label in Y.iloc[:, 0].unique()])
        self._fitClassifier(X, Y)

    def getClassLabels(self) -> List[Any]:
        return self._labels

    @abstractmethod
    def _fitClassifier(self, X: pd.DataFrame, y: pd.DataFrame):
        pass

    def convertClassProbabilitiesToPredictions(self, df: pd.DataFrame):
        """
        Converts from a result returned by predictClassProbabilities to a result as return by predict

        :param df: the output data frame from predictClassProbabilities
        :return: an output data frame as it would be returned by predict
        """
        dfCols = list(df.columns)
        if dfCols != self._labels:
            raise ValueError(f"Expected data frame with columns {self._labels}, got {dfCols}")
        yArray = df.values
        maxIndices = np.argmax(yArray, axis=1)
        result = [self._labels[i] for i in maxIndices]
        return pd.DataFrame(result, columns=self.getModelOutputVariableNames())

    def predictClassProbabilities(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        :param x: the input data
        :return: a data frame where the list of columns is the list of class labels and the values are probabilities
        """
        x = self._computeInputs(x)
        result = self._predictClassProbabilities(x)

        # check for correct columns
        if list(result.columns) != self._labels:
            raise Exception(f"_predictClassProbabilities returned DataFrame with incorrect columns: expected {self._labels}, got {result.columns}")

        # check for normalisation
        maxRowsToCheck = 5
        dfToCheck = result.iloc[:maxRowsToCheck]
        for i, (_, valueSeries) in enumerate(dfToCheck.iterrows(), start=1):
            s = valueSeries.sum()
            if abs(s-1.0) > 0.01:
                _log.warning(f"Probabilities data frame may not be correctly normalised: checked row {i}/{maxRowsToCheck} contains {list(valueSeries)}")

        return result

    @abstractmethod
    def _predictClassProbabilities(self, X: pd.DataFrame) -> pd.DataFrame:
        pass
