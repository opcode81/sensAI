from typing import Sequence, Tuple

import pandas as pd

from ..eval_stats.classification import VectorClassificationModelEvaluationData, \
    ClassificationEvalStats, VectorClassificationModelCrossValidationData
from ..evaluators.base import VectorModelEvaluator, VectorModelCrossValidator
from ...eval_stats import Metric
from ....data_ingest import InputOutputData
from ....models.vector_model import VectorClassificationModel


class VectorClassificationModelEvaluator(VectorModelEvaluator):
    def __init__(self, data: InputOutputData, testData: InputOutputData = None, dataSplitter=None, testFraction=None,
            randomSeed=42, computeProbabilities=False, shuffle=True, additionalMetrics: Sequence[Metric] = None):
        super().__init__(data=data, testData=testData, dataSplitter=dataSplitter, testFraction=testFraction, randomSeed=randomSeed, shuffle=shuffle)
        self.computeProbabilities = computeProbabilities
        self.additionalMetrics = additionalMetrics

    def evalModel(self, model: VectorClassificationModel, onTrainingData=False) -> VectorClassificationModelEvaluationData:
        if model.isRegressionModel():
            raise ValueError(f"Expected a classification model, got {model}")
        inputOutputData = self.trainingData if onTrainingData else self.testData
        predictions, predictions_proba, groundTruth = self._computeOutputs(model, inputOutputData)
        evalStats = ClassificationEvalStats(y_predictedClassProbabilities=predictions_proba, y_predicted=predictions, y_true=groundTruth,
            labels=model.getClassLabels(), additionalMetrics=self.additionalMetrics)
        predictedVarName = model.getPredictedVariableNames()[0]
        return VectorClassificationModelEvaluationData({predictedVarName: evalStats}, inputOutputData.inputs, model)

    def computeTestDataOutputs(self, model: VectorClassificationModel) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Applies the given model to the test data

        :param model: the model to apply
        :return: a triple (predictions, predicted class probability vectors, groundTruth) of DataFrames
        """
        return self._computeOutputs(model, self.testData)

    def _computeOutputs(self, model, inputOutputData: InputOutputData) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Applies the given model to the given data

        :param model: the model to apply
        :param inputOutputData: the data set
        :return: a triple (predictions, predicted class probability vectors, groundTruth) of DataFrames
        """
        if self.computeProbabilities:
            classProbabilities = model.predictClassProbabilities(inputOutputData.inputs)
            if classProbabilities is None:
                raise Exception(f"Requested computation of class probabilities for a model which does not support it: {model} returned None")
            predictions = model.convertClassProbabilitiesToPredictions(classProbabilities)
        else:
            classProbabilities = None
            predictions = model.predict(inputOutputData.inputs)
        groundTruth = inputOutputData.outputs
        return predictions, classProbabilities, groundTruth


class VectorClassificationModelCrossValidator(VectorModelCrossValidator[VectorClassificationModelCrossValidationData]):
    def _createModelEvaluator(self, trainingData: InputOutputData, testData: InputOutputData):
        return VectorClassificationModelEvaluator(trainingData, testData=testData, **self.evaluatorParams)

    def _createResultData(self, trainedModels, evalDataList, testIndicesList, predictedVarNames) -> VectorClassificationModelCrossValidationData:
        return VectorClassificationModelCrossValidationData(trainedModels, evalDataList, predictedVarNames)
