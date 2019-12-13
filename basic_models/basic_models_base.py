import copy
import logging
import time
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Sequence, List, Any

import numpy as np
import pandas as pd
import scipy.stats

from .eval_stats import RegressionEvalStats, EvalStats, ClassificationEvalStats, RegressionEvalStatsCollection, \
    ClassificationEvalStatsCollection, EvalStatsCollection

log = logging.getLogger(__name__)


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


class DataFrameTransformer(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class RuleBasedDataFrameTransformer(DataFrameTransformer, ABC):
    """Base class for transformers whose logic is enturely based on rules and does not need to be fitted to the data"""

    def fit(self, df: pd.DataFrame):
        pass


class DataFrameTransformerChain:
    def __init__(self, dataFrameTransformers: Sequence[DataFrameTransformer]):
        self.dataFrameTransformers = dataFrameTransformers

    def apply(self, df: pd.DataFrame, fit=False) -> pd.DataFrame:
        """
        Applies the chain of transformers to the given DataFrame, optionally fitting each transformer before applying it.
        Each transformer in the chain receives the transformed output of its predecessor.

        :param df: the data frame
        :param fit: whether to fit the transformers before applying them
        :return: the transformed data frame
        """
        for transformer in self.dataFrameTransformers:
            if fit:
                transformer.fit(df)
            df = transformer.apply(df)
        return df


class VectorModel(PredictorModel, ABC):
    """
    Base class for models that map vectors to vectors
    """

    def __init__(self, inputTransformers: Sequence[DataFrameTransformer] = (), outputTransformers: Sequence[DataFrameTransformer] = (),
            trainingOutputTransformers: Sequence[DataFrameTransformer] = ()):
        """
        :param inputTransformers: list of DataFrameTransformers for the transformation of inputs
        :param outputTransformers: list of DataFrameTransformers for the transformation of outputs
        :param trainingOutputTransformers: list of DataFrameTransformers for the transformation of training outputs prior to training
        """
        self._inputTransformerChain = DataFrameTransformerChain(inputTransformers)
        self._outputTransformerChain = DataFrameTransformerChain(outputTransformers)
        self._trainingOutputTransformerChain = DataFrameTransformerChain(trainingOutputTransformers)
        self._predictedVariableNames = None
        self._modelInputVariableNames = None
        self._modelOutputVariableNames = None

    @abstractmethod
    def isRegressionModel(self) -> bool:
        pass

    def _checkAndTransformInputs(self, x: pd.DataFrame):
        x = self._inputTransformerChain.apply(x)
        if self.getPredictedVariableNames() is None:
            raise Exception(f"Cannot obtain predictions from non-trained model {self.__class__}")
        if list(x.columns) != self._modelInputVariableNames:
            raise Exception(f"Inadmissible input data frame: expected columns {self._modelInputVariableNames}, got {list(x.columns)}")
        return x

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Performs a prediction for the given input data frame

        :param x: the input data
        :return: a DataFrame with the same index as the input
        """
        x = self._checkAndTransformInputs(x)
        y = self._predict(x)
        y.index = x.index
        y = self._outputTransformerChain.apply(y)
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
        self._predictedVariableNames = list(Y.columns)
        X = self._inputTransformerChain.apply(X, fit=True)
        Y = self._trainingOutputTransformerChain.apply(Y, fit=True)
        self._modelInputVariableNames = list(X.columns)
        self._modelOutputVariableNames = list(Y.columns)
        log.info(f"Training {self.__class__.__name__} with inputs={self._modelInputVariableNames}, outputs={list(Y.columns)}")
        self._fit(X, Y)

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


class VectorRegressionModel(VectorModel, ABC):
    def __init__(self, inputTransformers: Sequence[DataFrameTransformer] = (), outputTransformers: Sequence[DataFrameTransformer] = (),
            trainingOutputTransformers: Sequence[DataFrameTransformer] = ()):
        super().__init__(inputTransformers=inputTransformers, outputTransformers=outputTransformers,
            trainingOutputTransformers=trainingOutputTransformers)

    def isRegressionModel(self) -> bool:
        return True


class VectorClassificationModel(VectorModel, ABC):

    def __init__(self, inputTransformers=(), outputTransformers=(), trainingOutputTransformers=()):
        """
        Abstract base with prediction for class probabilities
        """
        self._labels = None
        super().__init__(inputTransformers=inputTransformers, outputTransformers=outputTransformers,
            trainingOutputTransformers=trainingOutputTransformers)

    def isRegressionModel(self) -> bool:
        return False

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        """
        Fits the model using the given data

        :param X: a data frame containing input data
        :param Y: a data frame containing output data
        """
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
        x = self._checkAndTransformInputs(x)
        return self._predictClassProbabilities(x)

    @abstractmethod
    def _predictClassProbabilities(self, X: pd.DataFrame) -> pd.DataFrame:
        pass


class VectorModelEvaluationData(ABC):
    @abstractmethod
    def getEvalStats(self) -> EvalStats:
        pass


class VectorRegressionModelEvaluationData(VectorModelEvaluationData):
    def __init__(self, statsDict: Dict[str, EvalStats]):
        """
        :param statsDict: a dictionary mapping from output variable name to the evaluation statistics object
        """
        self.data = statsDict

    def getEvalStats(self, predictedVarName=None):
        if predictedVarName is None:
            if len(self.data) != 1:
                raise Exception(f"Must provide name of predicted variable name, as multiple variables were predicted {list(self.data.keys())}")
            else:
                predictedVarName = next(iter(self.data.keys()))
        evalStats = self.data.get(predictedVarName)
        if evalStats is None:
            raise ValueError(f"No evaluation data present for '{predictedVarName}'; known output variables: {list(self.data.keys())}")
        return evalStats

    def getDataFrame(self):
        """
        Returns an DataFrame with all evaluation metrics (one row per output variable)

        :return: a DataFrame containing evaluation metrics
        """
        statsDicts = []
        varNames = []
        for predictedVarName, evalStats in self.data.items():
            statsDicts.append(evalStats.getAll())
            varNames.append(predictedVarName)
        df = pd.DataFrame(statsDicts, index=varNames)
        df.index.name = "predictedVar"
        return df


class VectorModelEvaluator(ABC):
    def __init__(self, data: InputOutputData, testFraction=None, testData: InputOutputData = None, randomSeed=42):
        """
        Constructs an evaluator with test and training data.
        Exactly one of the parameters {testFraction, testData} must be given

        :param data: the full data set, or, if testData is given, the training data
        :param testFraction: the fraction of the data to use for testing/evaluation
        :param testData: the data to use for testing/evaluation
        :param randomSeed: the random seed to use for splits of the data
        """
        self.testFraction = testFraction

        if self.testFraction is None and testData is None:
            raise Exception("Either testFraction or testData must be provided")
        if self.testFraction is not None and testData is not None:
            raise Exception("Cannot provide both testFraction and testData")

        if self.testFraction is not None:
            if not 0 <= self.testFraction <= 1:
                raise Exception(f"invalid testFraction: {testFraction}")
            numDataPoints = len(data)
            permutedIndices = np.random.RandomState(randomSeed).permutation(numDataPoints)
            splitIndex = int(numDataPoints * self.testFraction)
            trainingIndices = permutedIndices[splitIndex:]
            testIndices = permutedIndices[:splitIndex]
            self.trainingData = data.filterIndices(list(trainingIndices))
            self.testData = data.filterIndices(list(testIndices))
        else:
            self.trainingData = data
            self.testData = testData

    def fitModel(self, model: VectorModel):
        """Fits the given model's parameters using this evaluator's training data"""
        startTime = time.time()
        model.fit(self.trainingData.inputs, self.trainingData.outputs)
        log.info(f"Training of {model.__class__.__name__} completed in {time.time() - startTime:.1f} seconds")

    @abstractmethod
    def evalModel(self, model: PredictorModel) -> VectorModelEvaluationData:
        pass


class VectorRegressionModelEvaluator(VectorModelEvaluator):
    def __init__(self, data: InputOutputData, testFraction=None, testData: InputOutputData = None, randomSeed=42):
        super().__init__(data=data, testFraction=testFraction, testData=testData, randomSeed=randomSeed)

    def evalModel(self, model: PredictorModel) -> VectorRegressionModelEvaluationData:
        """
        :param model: the model to evaluate
        :return: a dictionary mapping from the predicted variable name to an object holding evaluation stats
        """
        if not model.isRegressionModel():
            raise ValueError("Expected regression model, got classification model instead")
        statsDict = {}
        predictions, groundTruth = self.computeTestDataOutputs(model)
        for predictedVarName in model.getPredictedVariableNames():
            evalStats = RegressionEvalStats(y_predicted=predictions[predictedVarName], y_true=groundTruth[predictedVarName])
            statsDict[predictedVarName] = evalStats
        return VectorRegressionModelEvaluationData(statsDict)

    def computeTestDataOutputs(self, model: PredictorModel) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies the given model to the test data

        :param model: the model to apply
        :return: a pair (predictions, groundTruth)
        """
        predictions = model.predict(self.testData.inputs)
        groundTruth = self.testData.outputs
        return predictions, groundTruth


class VectorClassificationModelEvaluationData(VectorModelEvaluationData):
    def __init__(self, evalStats: ClassificationEvalStats):
        self.evalStats = evalStats

    def getEvalStats(self) -> ClassificationEvalStats:
        return self.evalStats


class VectorClassificationModelEvaluator(VectorModelEvaluator):
    def __init__(self, data: InputOutputData, testFraction=None,
                 testData: InputOutputData = None, randomSeed=42, computeProbabilities=False):
        super().__init__(data=data, testFraction=testFraction, testData=testData, randomSeed=randomSeed)
        self.computeProbabilities = computeProbabilities

    def evalModel(self, model: VectorClassificationModel) -> VectorClassificationModelEvaluationData:
        if model.isRegressionModel():
            raise ValueError("Expected classification model, got regression model instead.")
        predictions, predictions_proba, groundTruth = self.computeTestDataOutputs(model)
        evalStats = ClassificationEvalStats(y_predictedClassProbabilities=predictions_proba, y_predicted=predictions, y_true=groundTruth, labels=model.getClassLabels())
        return VectorClassificationModelEvaluationData(evalStats)

    def computeTestDataOutputs(self, model: VectorClassificationModel) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Applies the given model to the test data

        :param model: the model to apply
        :return: a triple (predictions, predicted class probability vectors, groundTruth) of DataFrames
        """
        if self.computeProbabilities:
            classProbabilities = model.predictClassProbabilities(self.testData.inputs)
            predictions = model.convertClassProbabilitiesToPredictions(classProbabilities)
        else:
            classProbabilities = None
            predictions = model.predict(self.testData.inputs)
        groundTruth = self.testData.outputs
        return predictions, classProbabilities, groundTruth


class ChainedVectorRegressionPredictor(PredictorModel):
    def __init__(self, predictor: PredictorModel, nChainedPredictions: int):
        super().__init__()
        self.nChainedPredictions = nChainedPredictions
        self.predictor = predictor

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        nPredictions = 1
        predictions = self.predictor.predict(x)
        inputDim, outputDim = predictions.shape[1], x.shape[1]
        if inputDim != outputDim:
            raise Exception(f"Model {self.predictor.__class__} cannot be used for chained execution: "
                            f"inputDim {inputDim} does not match outputDim {outputDim}")
        while nPredictions < self.nChainedPredictions:
            predictions = self.predictor.predict(predictions)
            nPredictions += 1
        return predictions

    def getPredictedVariableNames(self):
        return self.predictor.getPredictedVariableNames()


class VectorModelCrossValidationData(ABC):
    @abstractmethod
    def getEvalStatsCollection(self) -> EvalStatsCollection:
        pass


class VectorModelCrossValidator(ABC):
    def __init__(self, data: InputOutputData, folds: int, randomSeed=42, returnTrainedModels=False):
        """
        :param data: the data set
        :param folds: the number of folds
        :param randomSeed: the random seed to use
        :param returnTrainedModels: whether to create a copy of the model for each fold and return each of the models
            (requires that models can be deep-copied); if False, the model that is passed to evalModel is fitted several times
        """
        self.returnTrainedModels = returnTrainedModels
        numDataPoints = len(data)
        permutedIndices = np.random.RandomState(randomSeed).permutation(numDataPoints)
        numTestPoints = numDataPoints // folds
        self.modelEvaluators = []
        for i in range(folds):
            testStartIdx = i * numTestPoints
            testEndIdx = testStartIdx + numTestPoints
            testIndices = permutedIndices[testStartIdx:testEndIdx]
            trainIndices = np.concatenate((permutedIndices[:testStartIdx], permutedIndices[testEndIdx:]))
            self.modelEvaluators.append(self._createModelEvaluator(data.filterIndices(trainIndices), data.filterIndices(testIndices)))

    @abstractmethod
    def _createModelEvaluator(self, trainingData: InputOutputData, testData: InputOutputData):
        pass

    def _evalModel(self, model):
        trainedModels = [] if self.returnTrainedModels else None
        evalDataList = []
        testIndicesList = []
        predictedVarNames = None
        for evaluator in self.modelEvaluators:
            modelToFit: VectorModel = copy.deepcopy(model) if self.returnTrainedModels else model
            evaluator.fitModel(modelToFit)
            if predictedVarNames is None:
                predictedVarNames = modelToFit.getPredictedVariableNames()
            if self.returnTrainedModels:
                trainedModels.append(modelToFit)
            evalDataList.append(evaluator.evalModel(modelToFit))
            testIndicesList.append(evaluator.testData.outputs.index)
        return trainedModels, evalDataList, testIndicesList, predictedVarNames

    @abstractmethod
    def evalModel(self, model) -> VectorModelCrossValidationData:
        pass


class VectorRegressionModelCrossValidationData(VectorModelCrossValidationData):
    def __init__(self, trainedModels, evalDataList, predictedVarNames, testIndicesList):
        self.predictedVarNames = predictedVarNames
        self.trainedModels = trainedModels
        self.evalDataList = evalDataList
        self.testIndicesList = testIndicesList

    def getEvalStatsCollection(self, predictedVarName=None) -> RegressionEvalStatsCollection:
        if predictedVarName is None:
            if len(self.predictedVarNames) != 1:
                raise Exception("Must provide name of predicted variable")
            else:
                predictedVarName = self.predictedVarNames[0]
        evalStatsList = [evalData.getEvalStats(predictedVarName) for evalData in self.evalDataList]
        return RegressionEvalStatsCollection(evalStatsList)


class VectorRegressionModelCrossValidator(VectorModelCrossValidator):
    def __init__(self, data: InputOutputData, folds=5, randomSeed=42, returnTrainedModels=False):
        super().__init__(data, folds=folds, randomSeed=randomSeed, returnTrainedModels=returnTrainedModels)

    @classmethod
    def _createModelEvaluator(cls, trainingData: InputOutputData, testData: InputOutputData):
        return VectorRegressionModelEvaluator(trainingData, testData=testData)

    def evalModel(self, model) -> VectorRegressionModelCrossValidationData:
        trainedModels, evalDataList, testIndicesList, predictedVarNames = self._evalModel(model)
        return VectorRegressionModelCrossValidationData(trainedModels, evalDataList, predictedVarNames, testIndicesList)


class VectorClassificationModelCrossValidationData(VectorModelCrossValidationData):
    def __init__(self, trainedModels, evalDataList: Sequence[VectorClassificationModelEvaluationData]):
        self.trainedModels = trainedModels
        self.evalDataList = evalDataList

    def getEvalStatsCollection(self) -> ClassificationEvalStatsCollection:
        evalStatsList = [evalData.getEvalStats() for evalData in self.evalDataList]
        return ClassificationEvalStatsCollection(evalStatsList)


class VectorClassificationModelCrossValidator(VectorModelCrossValidator):
    def __init__(self, data: InputOutputData, folds=5, randomSeed=42):
        super().__init__(data, folds=folds, randomSeed=randomSeed)

    @classmethod
    def _createModelEvaluator(cls, trainingData: InputOutputData, testData: InputOutputData):
        return VectorClassificationModelEvaluator(trainingData, testData=testData)

    def evalModel(self, model) -> VectorClassificationModelCrossValidationData:
        trainedModels, evalDataList, testIndicesList, predictedVarNames = self._evalModel(model)
        return VectorClassificationModelCrossValidationData(trainedModels, evalDataList)
