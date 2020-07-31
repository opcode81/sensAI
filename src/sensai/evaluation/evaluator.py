import logging
import time
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Generator, Generic, TypeVar, Sequence

import pandas as pd

from .eval_stats.eval_stats_base import EvalStats, EvalStatsCollection
from .eval_stats.eval_stats_classification import ClassificationEvalStats, ClassificationMetric
from .eval_stats.eval_stats_regression import RegressionEvalStats, RegressionEvalStatsCollection, RegressionMetric
from ..data_ingest import DataSplitter, DataSplitterFractional, InputOutputData
from ..util.typing import PandasNamedTuple
from ..vector_model import VectorClassificationModel, VectorModel, PredictorModel

log = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=VectorModel)
TEvalStats = TypeVar("TEvalStats", bound=EvalStats)
TEvalStatsCollection = TypeVar("TEvalStatsCollection", bound=EvalStatsCollection)


class MetricsDictProvider(ABC):
    @abstractmethod
    def computeMetrics(self, model, **kwargs) -> Dict[str, float]:
        pass


class PredictorModelEvaluationData(ABC, Generic[TEvalStats]):
    def __init__(self, statsDict: Dict[str, TEvalStats], inputData: pd.DataFrame, model: PredictorModel):
        """
        :param statsDict: a dictionary mapping from output variable name to the evaluation statistics object
        :param inputData: the input data that was used to produce the results
        :param model: the model that was used to produce predictions
        """
        self.inputData = inputData
        self.evalStatsByVarName = statsDict
        self.predictedVarNames = list(self.evalStatsByVarName.keys())
        self.modelName = model.getName()

    def getEvalStats(self, predictedVarName=None) -> TEvalStats:
        if predictedVarName is None:
            if len(self.evalStatsByVarName) != 1:
                raise Exception(f"Must provide name of predicted variable name, as multiple variables were predicted {list(self.evalStatsByVarName.keys())}")
            else:
                predictedVarName = next(iter(self.evalStatsByVarName.keys()))
        evalStats = self.evalStatsByVarName.get(predictedVarName)
        if evalStats is None:
            raise ValueError(f"No evaluation data present for '{predictedVarName}'; known output variables: {list(self.evalStatsByVarName.keys())}")
        return evalStats

    def getDataFrame(self):
        """
        Returns an DataFrame with all evaluation metrics (one row per output variable)

        :return: a DataFrame containing evaluation metrics
        """
        statsDicts = []
        varNames = []
        for predictedVarName, evalStats in self.evalStatsByVarName.items():
            statsDicts.append(evalStats.getAll())
            varNames.append(predictedVarName)
        df = pd.DataFrame(statsDicts, index=varNames)
        df.index.name = "predictedVar"
        return df

    def iterInputOutputGroundTruthTuples(self, predictedVarName=None) -> Generator[Tuple[PandasNamedTuple, Any, Any], None, None]:
        evalStats = self.getEvalStats(predictedVarName)
        for i, namedTuple in enumerate(self.inputData.itertuples()):
            yield namedTuple, evalStats.y_predicted[i], evalStats.y_true[i]


class VectorRegressionModelEvaluationData(PredictorModelEvaluationData[RegressionEvalStats]):
    def getEvalStatsCollection(self):
        return RegressionEvalStatsCollection(list(self.evalStatsByVarName.values()))


class VectorModelEvaluator(MetricsDictProvider, ABC):
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
        log.info(f"Training of {model.__class__.__name__} completed in {time.time() - startTime:.1f} seconds")

    @abstractmethod
    def evalModel(self, model: PredictorModel, onTrainingData=False) -> PredictorModelEvaluationData:
        """
        Evaluates the given model

        :param model: the model to evaluate
        :param onTrainingData: if True, evaluate on this evaluator's training data rather than the held-out test data
        :return: the evaluation result
        """
        pass

    def computeMetrics(self, model: PredictorModel, onTrainingData=False) -> Dict[str, float]:
        evalData = self.evalModel(model)
        return evalData.getEvalStats().getAll()


class VectorRegressionModelEvaluator(VectorModelEvaluator):
    def __init__(self, data: InputOutputData, testData: InputOutputData = None, dataSplitter=None, testFraction=None, randomSeed=42, shuffle=True,
            additionalMetrics: Sequence[RegressionMetric] = None):
        super().__init__(data=data, dataSplitter=dataSplitter, testFraction=testFraction, testData=testData, randomSeed=randomSeed, shuffle=shuffle)
        self.additionalMetrics = additionalMetrics

    def evalModel(self, model: PredictorModel, onTrainingData=False) -> VectorRegressionModelEvaluationData:
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

    def computeTestDataOutputs(self, model: PredictorModel) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


class VectorClassificationModelEvaluationData(PredictorModelEvaluationData[ClassificationEvalStats]):
    pass


class VectorClassificationModelEvaluator(VectorModelEvaluator):
    def __init__(self, data: InputOutputData, testData: InputOutputData = None, dataSplitter=None, testFraction=None,
            randomSeed=42, computeProbabilities=False, shuffle=True, additionalMetrics: Sequence[ClassificationMetric] = None):
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
