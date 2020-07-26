import copy
import logging
import time
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from ..eval_stats.base import VectorModelEvalStatsCollection, VectorModelEvaluationData
from ...evaluators import ModelEvaluator
from ....data_ingest import InputOutputData, DataSplitter, DataSplitterFractional
from ....models.vector_model import VectorModel

log = logging.getLogger(__name__)
TCrossValData = TypeVar("TCrossValData", bound="VectorModelCrossValidationData")
TModel = TypeVar("TModel", bound=VectorModel)
TEvalData = TypeVar("TEvalData", bound="VectorModelEvaluationData")
TEvalStatsCollection = TypeVar("TEvalStatsCollection", bound=VectorModelEvalStatsCollection)


class VectorModelEvaluator(ModelEvaluator, ABC):
    def __init__(self, data: InputOutputData, testData: InputOutputData = None, dataSplitter: DataSplitter = None,
            testFraction=None, randomSeed=42, shuffle=True):
        """
        Constructs an evaluator with test and training data.
        Exactly one of the parameters {testFraction, testData, } must be given

        :param data: the full data set, or, if testData is given, the training data
        :param testData: the data to use for testing/evaluation; if None, must specify either dataSplitter testFraction or dataSplitter
        :param dataSplitter: [if testData is None] a splitter to use in order to obtain; if None, must specify either testData or testFraction
        :param testFraction: [if testData is None, dataSplitter is None] the fraction of the data to use for testing/evaluation;
            if None, must specify either testData or dataSplitter
        :param randomSeed: [if data is None, dataSplitter is None] the random seed to use for the fractional split of the data
        :param shuffle: [if data is None, dataSplitter is None] whether to randomly (based on randomSeed) shuffle the dataset before
            splitting it
        """
        if (testData, dataSplitter, testFraction).count(None) != 2:
            raise ValueError("Exactly one of {testData, dataSplitter, testFraction} must be given")
        if testData is None:
            if dataSplitter is None:
                dataSplitter = DataSplitterFractional(1 - testFraction, shuffle=shuffle, randomSeed=randomSeed)
            self.trainingData, self.testData = dataSplitter.split(data)
        else:
            self.trainingData = data
            self.testData = testData

    def fitModel(self, model: VectorModel):
        """Fits the given model's parameters using this evaluator's training data"""
        startTime = time.time()
        model.fit(self.trainingData.inputs, self.trainingData.outputs)
        log.info(f"Training of {model.getName()} completed in {time.time() - startTime:.1f} seconds")

    @abstractmethod
    def evalModel(self, model: VectorModel, onTrainingData=False) -> VectorModelEvaluationData:
        """
        Evaluates the given model

        :param model: the model to evaluate
        :param onTrainingData: if True, evaluate on this evaluator's training data rather than the held-out test data
        :return: the evaluation result
        """
        pass


class VectorModelCrossValidator(ABC, Generic[TCrossValData]):
    def __init__(self, data: InputOutputData, folds: int = 5, randomSeed=42, returnTrainedModels=False, evaluatorParams: dict = None):
        """
        :param data: the data set
        :param folds: the number of folds
        :param randomSeed: the random seed to use
        :param returnTrainedModels: whether to create a copy of the model for each fold and return each of the models
            (requires that models can be deep-copied); if False, the model that is passed to evalModel is fitted several times
        :param evaluatorParams: keyword parameters with which to instantiate model evaluators
        """
        self.returnTrainedModels = returnTrainedModels
        self.evaluatorParams = evaluatorParams if evaluatorParams is not None else {}
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

    @abstractmethod
    def _createResultData(self, trainedModels, evalDataList, testIndicesList, predictedVarNames) -> TCrossValData:
        pass

    def evalModel(self, model):
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
        return self._createResultData(trainedModels, evalDataList, testIndicesList, predictedVarNames)
