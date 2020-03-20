import copy
import logging
import time

import matplotlib.figure
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Union, Generator, Generic, TypeVar, List, Optional, Sequence, Callable

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .util.io import ResultWriter
from .util.typing import PandasNamedTuple
from .vector_model import InputOutputData, VectorModel, PredictorModel, VectorClassificationModel, VectorRegressionModel
from .eval_stats import RegressionEvalStats, ClassificationEvalStats, RegressionEvalStatsCollection, \
    ClassificationEvalStatsCollection


_log = logging.getLogger(__name__)

TModel = TypeVar("TModel", VectorRegressionModel, VectorClassificationModel)
TEvalStats = TypeVar("TEvalStats", RegressionEvalStats, ClassificationEvalStats)
TEvalStatsCollection = TypeVar("TEvalStatsCollection", RegressionEvalStatsCollection, ClassificationEvalStatsCollection)
TEvaluator = TypeVar("TEvaluator", "VectorRegressionModelEvaluator", "VectorClassificationModelEvaluator")
TEvalData = TypeVar("TEvalData", "VectorRegressionModelEvaluationData", "VectorClassificationModelEvaluationData")
TCrossValData = TypeVar("TCrossValData", "VectorClassificationModelCrossValidationData", "VectorRegressionModelCrossValidationData")


class VectorModelEvaluationData(ABC, Generic[TEvalStats]):
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


class VectorRegressionModelEvaluationData(VectorModelEvaluationData[RegressionEvalStats]):
    def getEvalStatsCollection(self):
        return RegressionEvalStatsCollection(list(self.evalStatsByVarName.values()))


class VectorModelEvaluator(ABC):
    @staticmethod
    def forModel(model: VectorModel, data: InputOutputData, **kwargs) -> Union["VectorRegressionModelEvaluator", "VectorClassificationModelEvaluator"]:
        return VectorModelEvaluator.forModelType(model.isRegressionModel(), data, **kwargs)

    @staticmethod
    def forModelType(isRegression: bool, data: InputOutputData, **kwargs) -> Union["VectorRegressionModelEvaluator", "VectorClassificationModelEvaluator"]:
        if isRegression:
            return VectorRegressionModelEvaluator(data, **kwargs)
        else:
            return VectorClassificationModelEvaluator(data, **kwargs)

    def __init__(self, data: InputOutputData, testFraction=None, testData: InputOutputData = None, randomSeed=42, shuffle=True):
        """
        Constructs an evaluator with test and training data.
        Exactly one of the parameters {testFraction, testData} must be given

        :param data: the full data set, or, if testData is given, the training data
        :param testFraction: the fraction of the data to use for testing/evaluation
        :param testData: the data to use for testing/evaluation
        :param randomSeed: the random seed to use for splits of the data
        :param shuffle: whether to randomly (based on randomSeed) shuffle the dataset when splitting it
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
            splitIndex = int(numDataPoints * self.testFraction)
            if shuffle:
                indices = np.random.RandomState(randomSeed).permutation(numDataPoints)
            else:
                indices = range(numDataPoints)
            trainingIndices = indices[splitIndex:]
            testIndices = indices[:splitIndex]
            self.trainingData = data.filterIndices(list(trainingIndices))
            self.testData = data.filterIndices(list(testIndices))
        else:
            self.trainingData = data
            self.testData = testData

    def fitModel(self, model: VectorModel):
        """Fits the given model's parameters using this evaluator's training data"""
        startTime = time.time()
        model.fit(self.trainingData.inputs, self.trainingData.outputs)
        _log.info(f"Training of {model.__class__.__name__} completed in {time.time() - startTime:.1f} seconds")

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
    def __init__(self, data: InputOutputData, testFraction=None, testData: InputOutputData = None, randomSeed=42, shuffle=True):
        super().__init__(data=data, testFraction=testFraction, testData=testData, randomSeed=randomSeed, shuffle=shuffle)

    def evalModel(self, model: PredictorModel, onTrainingData=False) -> VectorRegressionModelEvaluationData:
        if not model.isRegressionModel():
            raise ValueError(f"Expected a regression model, got {model}")
        evalStatsByVarName = {}
        inputOutputData = self.trainingData if onTrainingData else self.testData
        predictions, groundTruth = self._computeOutputs(model, inputOutputData)
        for predictedVarName in model.getPredictedVariableNames():
            evalStats = RegressionEvalStats(y_predicted=predictions[predictedVarName], y_true=groundTruth[predictedVarName])
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


class VectorClassificationModelEvaluationData(VectorModelEvaluationData[ClassificationEvalStats]):
    pass


class VectorClassificationModelEvaluator(VectorModelEvaluator):
    def __init__(self, data: InputOutputData, testFraction=None, testData: InputOutputData = None,
            randomSeed=42, computeProbabilities=False, shuffle=True):
        super().__init__(data=data, testFraction=testFraction, testData=testData, randomSeed=randomSeed, shuffle=shuffle)
        self.computeProbabilities = computeProbabilities

    def evalModel(self, model: VectorClassificationModel, onTrainingData=False) -> VectorClassificationModelEvaluationData:
        if model.isRegressionModel():
            raise ValueError(f"Expected a classification model, got {model}")
        inputOutputData = self.trainingData if onTrainingData else self.testData
        predictions, predictions_proba, groundTruth = self._computeOutputs(model, inputOutputData)
        evalStats = ClassificationEvalStats(y_predictedClassProbabilities=predictions_proba, y_predicted=predictions, y_true=groundTruth, labels=model.getClassLabels())
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


class VectorModelCrossValidationData(ABC, Generic[TModel, TEvalData, TEvalStats, TEvalStatsCollection]):
    def __init__(self, trainedModels: List[TModel], evalDataList: List[TEvalData], predictedVarNames: List[str], testIndicesList=None):
        self.predictedVarNames = predictedVarNames
        self.trainedModels = trainedModels
        self.evalDataList = evalDataList
        self.testIndicesList = testIndicesList

    @property
    def modelName(self):
        return self.evalDataList[0].modelName

    @abstractmethod
    def _createEvalStatsCollection(self, l: List[TEvalStats]) -> TEvalStatsCollection:
        pass

    def getEvalStatsCollection(self, predictedVarName=None) -> TEvalStatsCollection:
        if predictedVarName is None:
            if len(self.predictedVarNames) != 1:
                raise Exception("Must provide name of predicted variable")
            else:
                predictedVarName = self.predictedVarNames[0]
        evalStatsList = [evalData.getEvalStats(predictedVarName) for evalData in self.evalDataList]
        return self._createEvalStatsCollection(evalStatsList)

    def iterInputOutputGroundTruthTuples(self, predictedVarName=None) -> Generator[Tuple[PandasNamedTuple, Any, Any], None, None]:
        for evalData in self.evalDataList:
            evalStats = evalData.getEvalStats(predictedVarName)
            for i, namedTuple in enumerate(evalData.inputData.itertuples()):
                yield namedTuple, evalStats.y_predicted[i], evalStats.y_true[i]


class VectorModelCrossValidator(ABC, Generic[TCrossValData]):
    @staticmethod
    def forModel(model: VectorModel, data: InputOutputData, folds=5, **kwargs) -> Union["VectorRegressionModelCrossValidator", "VectorClassificationModelCrossValidator"]:
        return VectorModelCrossValidator.forModelType(model.isRegressionModel(), data, folds=folds, **kwargs)

    @staticmethod
    def forModelType(isRegression: bool, data: InputOutputData, folds=5, **kwargs) -> Union["VectorRegressionModelCrossValidator", "VectorClassificationModelCrossValidator"]:
        if isRegression:
            return VectorRegressionModelCrossValidator(data, folds=folds, **kwargs)
        else:
            return VectorClassificationModelCrossValidator(data, folds=folds, **kwargs)

    def __init__(self, data: InputOutputData, folds: int = 5, randomSeed=42, returnTrainedModels=False):
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


class VectorRegressionModelCrossValidationData(VectorModelCrossValidationData[VectorRegressionModel, VectorRegressionModelEvaluationData, RegressionEvalStats, RegressionEvalStatsCollection]):
    def _createEvalStatsCollection(self, l: List[RegressionEvalStats]) -> RegressionEvalStatsCollection:
        return RegressionEvalStatsCollection(l)


class VectorRegressionModelCrossValidator(VectorModelCrossValidator[VectorRegressionModelCrossValidationData]):
    def _createModelEvaluator(self, trainingData: InputOutputData, testData: InputOutputData) -> VectorRegressionModelEvaluator:
        return VectorRegressionModelEvaluator(trainingData, testData=testData)

    def _createResultData(self, trainedModels, evalDataList, testIndicesList, predictedVarNames) -> VectorRegressionModelCrossValidationData:
        return VectorRegressionModelCrossValidationData(trainedModels, evalDataList, predictedVarNames, testIndicesList)


class VectorClassificationModelCrossValidationData(VectorModelCrossValidationData[VectorClassificationModel, VectorClassificationModelEvaluationData, ClassificationEvalStats, ClassificationEvalStatsCollection]):
    def _createEvalStatsCollection(self, l: List[ClassificationEvalStats]) -> ClassificationEvalStatsCollection:
        return ClassificationEvalStatsCollection(l)


class VectorClassificationModelCrossValidator(VectorModelCrossValidator[VectorClassificationModelCrossValidationData]):
    def _createModelEvaluator(self, trainingData: InputOutputData, testData: InputOutputData):
        return VectorClassificationModelEvaluator(trainingData, testData=testData)

    def _createResultData(self, trainedModels, evalDataList, testIndicesList, predictedVarNames) -> VectorClassificationModelCrossValidationData:
        return VectorClassificationModelCrossValidationData(trainedModels, evalDataList, predictedVarNames)


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


def evalModelViaEvaluator(model: TModel, inputOutputData: InputOutputData, testFraction=0.2,
        plotTargetDistribution=False, computeProbabilities=True, normalizePlots=True, randomSeed=60) -> TEvalData:
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
        _log.info(f"Description of target column in training set: \n{outputDistributionSeries.describe()}")
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
        evaluatorParams = dict(testFraction=testFraction, randomSeed=randomSeed)
    else:
        evaluatorParams = dict(testFraction=testFraction, computeProbabilities=computeProbabilities, randomSeed=randomSeed)
    ev = EvaluationUtil.forModelType(model.isRegressionModel(), inputOutputData, evaluatorParams=evaluatorParams)
    return ev.performSimpleEvaluation(model, showPlots=True, logResults=True)


class EvaluationUtil(ABC, Generic[TModel, TEvaluator, TEvalData, TCrossValData, TEvalStats]):
    """
    Utility class for the evaluation of models based on a dataset
    """
    def __init__(self, inputOutputData: InputOutputData, evaluatorParams: Optional[Dict[str, Any]] = None,
            crossValidatorParams: Optional[Dict[str, Any]] = None):
        """
        :param inputOutputData: the data set to use for evaluation
        :param evaluatorParams: parameters with which to instantiate evaluators
        :param crossValidatorParams: parameters with which to instantiate cross-validators
        """
        if evaluatorParams is None:
            evaluatorParams = dict(testFraction=0.2)
        if crossValidatorParams is None:
            crossValidatorParams = dict(folds=5)
        self.evaluatorParams = evaluatorParams
        self.crossValidatorParams = crossValidatorParams
        self.inputOutputData = inputOutputData

    @staticmethod
    def forModelType(isRegression, inputOutputData: InputOutputData, evaluatorParams: Optional[Dict[str, Any]] = None,
            crossValidatorParams: Optional[Dict[str, Any]] = None) -> Union["ClassificationEvaluationUtil", "RegressionEvaluationUtil"]:
        cons = RegressionEvaluationUtil if isRegression else ClassificationEvaluationUtil
        return cons(inputOutputData, evaluatorParams=evaluatorParams, crossValidatorParams=crossValidatorParams)

    class ResultCollector:
        def __init__(self, showPlots: bool = True, resultWriter: Optional[ResultWriter] = None):
            self.showPlots = showPlots
            self.resultWriter = resultWriter

        def addFigure(self, name, fig: matplotlib.figure.Figure):
            if self.resultWriter is not None:
                self.resultWriter.writeFigure(name, fig, closeFigure=not self.showPlots)

        def child(self, addedFilenamePrefix):
            resultWriter = self.resultWriter
            if resultWriter:
                resultWriter = resultWriter.childWithAddedPrefix(addedFilenamePrefix)
            return self.__class__(showPlots=self.showPlots, resultWriter=resultWriter)

    def createEvaluator(self, model) -> TEvaluator:
        """
        Creates an evaluator which is suitable for evaluation of the given model
        :param model: the model for which to create an evaluator
        :return: an evaluator
        """
        return VectorModelEvaluator.forModel(model, self.inputOutputData, **self.evaluatorParams)

    def performSimpleEvaluation(self, model: TModel, showPlots=False, logResults=True, resultWriter: ResultWriter = None) -> TEvalData:
        evaluator = self.createEvaluator(model)
        evaluator.fitModel(model)
        evaluator.evalModel(model)
        evalResultData = evaluator.evalModel(model)
        strEvalResults = str(evalResultData.getEvalStats())
        if logResults:
            _log.info(f"Evaluation results: {strEvalResults}")
        if resultWriter is not None:
            resultWriter.writeTextFile("evaluator-results", strEvalResults)
        self.createPlots(evalResultData, showPlots=showPlots, resultWriter=resultWriter)
        return evalResultData

    @staticmethod
    def _resultWriterForModel(resultWriter: Optional[ResultWriter], model: TModel) -> Optional[ResultWriter]:
        if resultWriter is None:
            return None
        return resultWriter.childWithAddedPrefix(model.getName() + "-")

    def performCrossValidation(self, model: TModel, showPlots=False, logResults=True, resultWriter: Optional[ResultWriter] = None) -> TCrossValData:
        """
        Evaluates the given model via cross-validation

        :param model: the model to evaluate
        :param showPlots: whether to show plots that visualise evaluation results (combining all folds)
        :param logResults: whether to log evaluation results
        :param resultWriter: a writer with which to store text files and plots. The evaluated model's name is added to each filename
            automatically
        :return: cross-validation result data
        """
        resultWriter = self._resultWriterForModel(resultWriter, model)
        crossValidator = VectorModelCrossValidator.forModel(model, self.inputOutputData, **self.crossValidatorParams)
        crossValidationData = crossValidator.evalModel(model)
        strEvalResults = str(crossValidationData.getEvalStatsCollection().aggStats())
        if logResults:
            _log.info(f"Cross-validation results: {strEvalResults}")
        if resultWriter is not None:
            resultWriter.writeTextFile("evaluator-results", strEvalResults)
        self.createPlots(crossValidationData, showPlots=showPlots, resultWriter=resultWriter)
        return crossValidationData

    def compareModelsCrossValidation(self, models: Sequence[TModel], resultWriter: Optional[ResultWriter] = None) -> pd.DataFrame:
        """
        Compares several models via cross-validation

        :param models: the models to compare
        :param resultWriter: a writer with which to store results of the comparison
        :return: a data frame containing evaluation metrics on all models
        """
        statsList = []
        for model in models:
            crossValidationResult = self.performCrossValidation(model, resultWriter=resultWriter)
            stats = crossValidationResult.getEvalStatsCollection().aggStats()
            stats["modelName"] = model.getName()
            statsList.append(stats)
        resultsDF = pd.DataFrame(statsList).set_index("modelName")
        strResults = f"Model comparison results:\n{resultsDF.to_string()}"
        _log.info(strResults)
        if resultWriter is not None:
            resultWriter.writeTextFile("model-comparison-results", strResults)
        return resultsDF

    def createPlots(self, data: Union[TEvalData, TCrossValData], showPlots=True, resultWriter: Optional[ResultWriter] = None, subtitle: str = None):
        """
        Creates default plots that visualise the results in the given evaluation data

        :param data: the evaluation data for which to create the default plots
        :param predictedVarName: the predicted variable for which to create plots; may be None if there is only one
        :param showPlots: whether to show plots
        :param resultWriter: if not None, plots will be written using this writer
        :param subtitle: the subtitle to use in plot titles (if any)
        """
        if not showPlots and resultWriter is None:
            return
        resultCollector = self.ResultCollector(showPlots=showPlots, resultWriter=resultWriter)
        self._createPlots(data, resultCollector, subtitle=data.modelName)

    def _createPlots(self, data: Union[TEvalData, TCrossValData], resultCollector: ResultCollector, subtitle=None):

        def createPlots(predVarName, rc):
            if isinstance(data, VectorModelCrossValidationData):
                evalStats = data.getEvalStatsCollection(predictedVarName=predVarName).getGlobalStats()
            elif isinstance(data, VectorModelEvaluationData):
                evalStats = data.getEvalStats(predictedVarName=predVarName)
            else:
                raise ValueError(f"Unexpected argument: data={data}")
            return self._createEvalStatsPlots(evalStats, rc, subtitle=subtitle)

        predictedVarNames = data.predictedVarNames
        if len(predictedVarNames) == 1:
            createPlots(predictedVarNames[0], resultCollector)
        else:
            for predictedVarName in predictedVarNames:
                createPlots(predictedVarName, resultCollector.child(predictedVarName+"-"))

    @abstractmethod
    def _createEvalStatsPlots(self, evalStats: TEvalStats, resultCollector: ResultCollector, subtitle=None):
        """
        :param evalStats: the evaluation results for which to create plots
        :param resultCollector: the collector to which all plots are to be passed
        :param subtitle: the subtitle to use for generated plots (if any)
        """
        pass


class RegressionEvaluationUtil(EvaluationUtil[VectorRegressionModel, VectorRegressionModelEvaluator, VectorRegressionModelEvaluationData, VectorRegressionModelCrossValidationData, RegressionEvalStats]):
    def _createEvalStatsPlots(self, evalStats: RegressionEvalStats, resultCollector: EvaluationUtil.ResultCollector, subtitle=None):
        resultCollector.addFigure("error-dist", evalStats.plotErrorDistribution(titleAdd=subtitle))
        resultCollector.addFigure("heatmap-gt-pred", evalStats.plotHeatmapGroundTruthPredictions(titleAdd=subtitle))
        resultCollector.addFigure("scatter-gt-pred", evalStats.plotScatterGroundTruthPredictions(titleAdd=subtitle))


class ClassificationEvaluationUtil(EvaluationUtil[VectorClassificationModel, VectorClassificationModelEvaluator, VectorClassificationModelEvaluationData, VectorClassificationModelCrossValidationData, ClassificationEvalStats]):
    def _createEvalStatsPlots(self, evalStats: ClassificationEvalStats, resultCollector: EvaluationUtil.ResultCollector, subtitle=None):
        resultCollector.addFigure("confusion-matrix", evalStats.plotConfusionMatrix(titleAdd=subtitle))


class MultiDataEvaluationUtil:
    def __init__(self, inputOutputDataDict: Dict[str, InputOutputData], keyName: str = "dataset"):
        """
        :param inputOutputDataDict: a dictionary mapping from names to the data sets with which to evaluate models
        :param keyName: a name for the key value used in inputOutputDataDict
        """
        self.inputOutputDataDict = inputOutputDataDict
        self.keyName = keyName

    def compareModelsCrossValidation(self, isRegression, modelFactories: Sequence[Callable[[], VectorModel]],
            resultWriter: Optional[ResultWriter] = None, crossValidatorParams: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        :param isRegression: flag indicating whether the models to evaluate are regression models
        :param modelFactories: a sequence of factory functions for the creation of models to evaluate
        :param resultWriter: a writer with which to store results
        :param crossValidatorParams: parameters to use for the instantiation of cross-validators
        :return: a pair of data frames (allDF, meanDF) where allDF contains all the individual cross-validation results
            for every dataset and meanDF contains one row for each model with results averaged across datasets
        """
        allResults = pd.DataFrame()
        for key, inputOutputData in self.inputOutputDataDict.items():
            _log.info(f"Evaluating models for {key}")
            ev = EvaluationUtil.forModelType(isRegression, inputOutputData, crossValidatorParams=crossValidatorParams)
            models = [f() for f in modelFactories]
            df = ev.compareModelsCrossValidation(models)
            df[self.keyName] = key
            df["modelName"] = df.index
            df = df.reset_index(drop=True)
            allResults = pd.concat((allResults, df))
        strAllResults = f"All results:\n{allResults.to_string()}"
        _log.info(strAllResults)
        meanResults = allResults.groupby("modelName").mean()
        strMeanResults = f"Mean results:\n{meanResults.to_string()}"
        _log.info(strMeanResults)
        if resultWriter is not None:
            resultWriter.writeTextFile("model-comparison-results", strMeanResults + "\n\n" + strAllResults)
        return allResults, meanResults