import copy
import logging
import time
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Sequence, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .vector_model import InputOutputData, VectorModel, PredictorModel, VectorClassificationModel
from .eval_stats import RegressionEvalStats, EvalStats, ClassificationEvalStats, RegressionEvalStatsCollection, \
    ClassificationEvalStatsCollection, EvalStatsCollection

log = logging.getLogger(__name__)


class VectorModelEvaluationData(ABC):
    @abstractmethod
    def getEvalStats(self) -> EvalStats:
        pass


class VectorRegressionModelEvaluationData(VectorModelEvaluationData):
    def __init__(self, statsDict: Dict[str, RegressionEvalStats]):
        """
        :param statsDict: a dictionary mapping from output variable name to the evaluation statistics object
        """
        self.data = statsDict

    def getEvalStats(self, predictedVarName=None) -> RegressionEvalStats:
        if predictedVarName is None:
            if len(self.data) != 1:
                raise Exception(f"Must provide name of predicted variable name, as multiple variables were predicted {list(self.data.keys())}")
            else:
                predictedVarName = next(iter(self.data.keys()))
        evalStats = self.data.get(predictedVarName)
        if evalStats is None:
            raise ValueError(f"No evaluation data present for '{predictedVarName}'; known output variables: {list(self.data.keys())}")
        return evalStats

    def getEvalStatsCollection(self):
        return RegressionEvalStatsCollection(list(self.data.values()))

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
    @staticmethod
    def forModel(model: VectorModel, data: InputOutputData, **kwargs) -> "VectorModelEvaluator":
        if model.isRegressionModel():
            return VectorRegressionModelEvaluator(data, **kwargs)
        else:
            return VectorClassificationModelEvaluator(data, **kwargs)

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
    def evalModel(self, model: PredictorModel, onTrainingData=False) -> VectorModelEvaluationData:
        """
        Evaluates the given model

        :param model: the model to evaluate
        :param onTrainingData: if True, evaluate on this evaluator's training data rather than the held-out test data
        :return: the evaluation result
        """
        pass


class VectorRegressionModelEvaluator(VectorModelEvaluator):
    def __init__(self, data: InputOutputData, testFraction=None, testData: InputOutputData = None, randomSeed=42):
        super().__init__(data=data, testFraction=testFraction, testData=testData, randomSeed=randomSeed)

    def evalModel(self, model: PredictorModel, onTrainingData=False) -> VectorRegressionModelEvaluationData:
        if not model.isRegressionModel():
            raise ValueError(f"Expected a regression model, got {model}")
        statsDict = {}
        predictions, groundTruth = self.computeOutputs(model, self.trainingData if onTrainingData else self.testData)
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
        return self.computeOutputs(model, self.testData)

    @staticmethod
    def computeOutputs(model, inputOutputData: InputOutputData):
        """
        Applies the given model to the given data

        :param model: the model to apply
        :param inputOutputData: the data set
        :return: a pair (predictions, groundTruth)
        """
        predictions = model.predict(inputOutputData.inputs)
        groundTruth = inputOutputData.outputs
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

    def evalModel(self, model: VectorClassificationModel, onTrainingData=False) -> VectorClassificationModelEvaluationData:
        if model.isRegressionModel():
            raise ValueError(f"Expected a classification model, got {model}")
        predictions, predictions_proba, groundTruth = self.computeOutputs(model, self.trainingData if onTrainingData else self.testData)
        evalStats = ClassificationEvalStats(y_predictedClassProbabilities=predictions_proba, y_predicted=predictions, y_true=groundTruth, labels=model.getClassLabels())
        return VectorClassificationModelEvaluationData(evalStats)

    def computeTestDataOutputs(self, model: VectorClassificationModel) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Applies the given model to the test data

        :param model: the model to apply
        :return: a triple (predictions, predicted class probability vectors, groundTruth) of DataFrames
        """
        return self.computeOutputs(model, self.testData)

    def computeOutputs(self, model, inputOutputData: InputOutputData) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Applies the given model to the given data

        :param model: the model to apply
        :param inputOutputData: the data set
        :return: a triple (predictions, predicted class probability vectors, groundTruth) of DataFrames
        """
        if self.computeProbabilities:
            classProbabilities = model.predictClassProbabilities(self.testData.inputs)
            if classProbabilities is None:
                raise Exception(f"Requested computation of class probabilities for a model which does not support it: {model} returned None")
            predictions = model.convertClassProbabilitiesToPredictions(classProbabilities)
        else:
            classProbabilities = None
            predictions = model.predict(self.testData.inputs)
        groundTruth = self.testData.outputs
        return predictions, classProbabilities, groundTruth


class VectorModelCrossValidationData(ABC):
    @abstractmethod
    def getEvalStatsCollection(self) -> EvalStatsCollection:
        pass


class VectorModelCrossValidator(ABC):
    @staticmethod
    def forModel(model: VectorModel, data: InputOutputData, folds=5, **kwargs) -> "VectorModelCrossValidator":
        if model.isRegressionModel():
            return VectorRegressionModelCrossValidator(data, folds=folds, **kwargs)
        else:
            return VectorClassificationModelCrossValidator(data, folds=folds, **kwargs)

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


def computeEvaluationMetricsDict(model, evaluatorOrValidator: Union[VectorModelEvaluator, VectorModelCrossValidator]) -> Dict[str, float]:
    if isinstance(evaluatorOrValidator, VectorModelEvaluator):
        evaluator: VectorModelEvaluator = evaluatorOrValidator
        evaluator.fitModel(model)
        data = evaluator.evalModel(model)
        return data.getEvalStats().getAll()
    elif isinstance(evaluatorOrValidator, VectorModelCrossValidator):
        crossValidator: VectorModelCrossValidator = evaluatorOrValidator
        data = crossValidator.evalModel(model)
        return data.getEvalStatsCollection().aggStats()
    else:
        raise ValueError(f"Unexpected evaluator/validator of type {type(evaluatorOrValidator)}")


def evalModelViaEvaluator(model: VectorModel, inputOutputData: InputOutputData, testFraction=0.2,
        plotTargetDistribution=False, computeProbabilities=True, normalizePlots=True, randomSeed=60) -> VectorModelEvaluationData:
    """
    Evaluates the given model via a simple evaluation mechanism that uses a single split

    :param model: the model to evaluate
    :param inputOutputData: data on which to evaluate
    :param testFraction: the fraction of the data to test on
    :param plotTargetDistribution: whether to plot the target values distribution in the entire dataset
    :param computeProbabilities: only relevant if the model is a classifier
    :param randomSeed:

    :return: the evaluation data
    """

    if plotTargetDistribution:
        title = "Distribution of target values in entire dataset"
        fig = plt.figure(title)

        outputDistributionSeries = inputOutputData.outputs.iloc[:, 0]
        log.info(f"Description of target column in training set: \n{outputDistributionSeries.describe()}")
        if not model.isRegressionModel():
            outputDistributionSeries = outputDistributionSeries.value_counts(normalize=normalizePlots)
            ax = sns.barplot(outputDistributionSeries.index, outputDistributionSeries.values)
            ax.set_ylabel("%")
        else:
            ax = sns.distplot(outputDistributionSeries)
            ax.set_ylabel("Probability density")
        ax.set_title(title)
        ax.set_xlabel("target value")
        fig.show()

    if model.isRegressionModel():
        evaluator = VectorRegressionModelEvaluator(inputOutputData, testFraction=testFraction, randomSeed=randomSeed)
    else:
        evaluator = VectorClassificationModelEvaluator(inputOutputData, testFraction=testFraction, computeProbabilities=computeProbabilities, randomSeed=randomSeed)

    tStart = time.time()
    evaluator.fitModel(model)
    evalData = evaluator.evalModel(model)
    evalStats = evalData.getEvalStats()
    log.info(f"Finished evaluation for model {model} in {time.time() - tStart} seconds")
    log.info(f"Evaluation metrics: {str(evalStats.getAll())}")

    if model.isRegressionModel():
        res: RegressionEvalStats = evalStats
        res.plotErrorDistribution()
        res.plotScatterGroundTruthPredictions()
        res.plotHeatmapGroundTruthPredictions()
    else:
        res: ClassificationEvalStats = evalStats
        res.plotConfusionMatrix(normalize=normalizePlots)

    return evalData
