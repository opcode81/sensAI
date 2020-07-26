from typing import Sequence, Tuple

import pandas as pd

from ..eval_stats.regression import VectorRegressionModelEvaluationData, \
    RegressionEvalStats, VectorRegressionModelCrossValidationData, RegressionMetric
from ..evaluators.base import VectorModelEvaluator, VectorModelCrossValidator
from ....data_ingest import InputOutputData
from ....models.vector_model import VectorModel


class VectorRegressionModelEvaluator(VectorModelEvaluator):
    def __init__(self, data: InputOutputData, testData: InputOutputData = None, dataSplitter=None,
            testFraction=None, randomSeed=42, shuffle=True, additionalMetrics: Sequence[RegressionMetric] = None):
        super().__init__(data=data, dataSplitter=dataSplitter, testFraction=testFraction, testData=testData,
                randomSeed=randomSeed, shuffle=shuffle)
        self.additionalMetrics = additionalMetrics

    def evalModel(self, model: VectorModel, onTrainingData=False) -> VectorRegressionModelEvaluationData:
        if not model.isRegressionModel():
            raise ValueError(f"Expected a regression model, got {model}")
        evalStatsByVarName = {}
        inputOutputData = self.trainingData if onTrainingData else self.testData
        predictions, groundTruth = self._computeOutputs(model, inputOutputData)
        for predictedVarName in model.getPredictedVariableNames():
            evalStats = RegressionEvalStats(y_predicted=predictions[predictedVarName], y_true=groundTruth[predictedVarName],
                additionalMetrics=self.additionalMetrics)
            evalStatsByVarName[predictedVarName] = evalStats
        return VectorRegressionModelEvaluationData(evalStatsByVarName, inputOutputData.inputs, model)

    def computeTestDataOutputs(self, model: VectorModel) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies the given model to the test data

        :param model: the model to apply
        :return: a pair (predictions, groundTruth)
        """
        return self._computeOutputs(model, self.testData)

    def _computeOutputs(self, model, inputOutputData: InputOutputData):
        """
        Applies the given model to the given data

        :param model: the model to apply
        :param inputOutputData: the data set
        :return: a pair (predictions, groundTruth)
        """
        predictions = model.predict(inputOutputData.inputs)
        groundTruth = inputOutputData.outputs
        return predictions, groundTruth


class VectorRegressionModelCrossValidator(VectorModelCrossValidator[VectorRegressionModelCrossValidationData]):
    def _createModelEvaluator(self, trainingData: InputOutputData, testData: InputOutputData) -> VectorRegressionModelEvaluator:
        return VectorRegressionModelEvaluator(trainingData, testData=testData, **self.evaluatorParams)

    def _createResultData(self, trainedModels, evalDataList, testIndicesList, predictedVarNames) -> VectorRegressionModelCrossValidationData:
        return VectorRegressionModelCrossValidationData(trainedModels, evalDataList, predictedVarNames, testIndicesList)
