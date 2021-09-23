import copy
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Any, Generator, Generic, TypeVar, List, Union, Sequence, Iterable

import numpy as np

from .eval_stats.eval_stats_base import PredictionEvalStats, EvalStatsCollection
from .eval_stats.eval_stats_classification import ClassificationEvalStats, ClassificationEvalStatsCollection
from .eval_stats.eval_stats_regression import RegressionEvalStats, RegressionEvalStatsCollection
from .evaluator import VectorRegressionModelEvaluationData, VectorClassificationModelEvaluationData, \
    VectorModelEvaluationData, VectorClassificationModelEvaluator, VectorRegressionModelEvaluator, \
    MetricsDictProvider
from ..data import InputOutputData
from ..util.typing import PandasNamedTuple
from ..vector_model import VectorClassificationModel, VectorRegressionModel, VectorModel

log = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=VectorModel)
TEvalStats = TypeVar("TEvalStats", bound=PredictionEvalStats)
TEvalStatsCollection = TypeVar("TEvalStatsCollection", bound=EvalStatsCollection)
TEvalData = TypeVar("TEvalData", bound=VectorModelEvaluationData)


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


TCrossValData = TypeVar("TCrossValData", bound=VectorModelCrossValidationData)


class CrossValidationSplitter(ABC):
    """
    Defines a mechanism with which to generate data splits for cross-validation
    """
    @abstractmethod
    def createFolds(self, data: InputOutputData, numFolds: int) -> List[Tuple[Sequence[int], Sequence[int]]]:
        """
        :param data: the data from which to obtain the folds
        :param numFolds: the number of splits/folds
        :return: a list containing numSplits tuples (t, e) where t and e are sequences of data point indices to use for training
            and evaluation respectively
        """
        pass


class CrossValidationSplitterDefault(CrossValidationSplitter):
    def __init__(self, shuffle=True, randomSeed=42):
        self.shuffle = shuffle
        self.randomSeed = randomSeed

    def createFolds(self, data: InputOutputData, numSplits: int) -> List[Tuple[Sequence[int], Sequence[int]]]:
        numDataPoints = len(data)
        numTestPoints = numDataPoints // numSplits
        if self.shuffle:
            indices = np.random.RandomState(self.randomSeed).permutation(numDataPoints)
        else:
            indices = list(range(numDataPoints))
        result = []
        for i in range(numSplits):
            testStartIdx = i * numTestPoints
            testEndIdx = testStartIdx + numTestPoints
            testIndices = indices[testStartIdx:testEndIdx]
            trainIndices = np.concatenate((indices[:testStartIdx], indices[testEndIdx:]))
            result.append((trainIndices, testIndices))
        return result


class VectorModelCrossValidator(MetricsDictProvider, Generic[TCrossValData], ABC):
    def __init__(self, data: InputOutputData, folds: int = 5, randomSeed=42, returnTrainedModels=False, evaluatorParams: dict = None,
            shuffle=True, splitter: CrossValidationSplitter = None):
        """
        :param data: the data set
        :param folds: the number of folds
        :param randomSeed: [if splitter is None] the random seed to use for splits
        :param returnTrainedModels: whether to create a copy of the model for each fold and return each of the models
            (requires that models can be deep-copied); if False, the model that is passed to evalModel is fitted several times
        :param evaluatorParams: keyword parameters with which to instantiate model evaluators
        :param shuffle: [if splitter is None] whether to shuffle the data (using randomSeed) before creating the folds
        :param splitter: the splitter to use in order to generate the folds; if None, use default split (using parameters randomSeed
            and shuffle above)
        """
        self.returnTrainedModels = returnTrainedModels
        self.evaluatorParams = evaluatorParams if evaluatorParams is not None else {}
        self.modelEvaluators = []
        if splitter is None:
            splitter = CrossValidationSplitterDefault(shuffle=shuffle, randomSeed=randomSeed)
        for trainIndices, testIndices in splitter.createFolds(data, folds):
            self.modelEvaluators.append(self._createModelEvaluator(data.filterIndices(trainIndices), data.filterIndices(testIndices)))

    @staticmethod
    def forModel(model: VectorModel, data: InputOutputData, folds=5, **kwargs) \
            -> Union["VectorClassificationModelCrossValidator", "VectorRegressionModelCrossValidator"]:
        cons = VectorRegressionModelCrossValidator if model.isRegressionModel() else VectorClassificationModelCrossValidator
        return cons(data, folds=folds, **kwargs)

    @abstractmethod
    def _createModelEvaluator(self, trainingData: InputOutputData, testData: InputOutputData):
        pass

    @abstractmethod
    def _createResultData(self, trainedModels, evalDataList, testIndicesList, predictedVarNames) -> TCrossValData:
        pass

    def evalModel(self, model: VectorModel):
        trainedModels = [] if self.returnTrainedModels else None
        evalDataList = []
        testIndicesList = []
        predictedVarNames = None
        for i, evaluator in enumerate(self.modelEvaluators, start=1):
            log.info(f"Training and evaluating model with fold {i}/{len(self.modelEvaluators)} ...")
            modelToFit: VectorModel = copy.deepcopy(model) if self.returnTrainedModels else model
            evaluator.fitModel(modelToFit)
            if predictedVarNames is None:
                predictedVarNames = modelToFit.getPredictedVariableNames()
            if self.returnTrainedModels:
                trainedModels.append(modelToFit)
            evalData = evaluator.evalModel(modelToFit)
            log.info(f"Evaluation result for fold {i}/{len(self.modelEvaluators)}: {evalData.getEvalStats()}")
            evalDataList.append(evalData)
            testIndicesList.append(evaluator.testData.outputs.index)
        return self._createResultData(trainedModels, evalDataList, testIndicesList, predictedVarNames)

    def _computeMetrics(self, model: VectorModel):
        data = self.evalModel(model)
        return data.getEvalStatsCollection().aggStats()


class VectorRegressionModelCrossValidationData(VectorModelCrossValidationData[VectorRegressionModel, VectorRegressionModelEvaluationData, RegressionEvalStats, RegressionEvalStatsCollection]):
    def _createEvalStatsCollection(self, l: List[RegressionEvalStats]) -> RegressionEvalStatsCollection:
        return RegressionEvalStatsCollection(l)


class VectorRegressionModelCrossValidator(VectorModelCrossValidator[VectorRegressionModelCrossValidationData]):
    # TODO: after switching to python3.8 we can move both methods to the base class by accessing the generic type at runtime
    def _createModelEvaluator(self, trainingData: InputOutputData, testData: InputOutputData) -> VectorRegressionModelEvaluator:
        return VectorRegressionModelEvaluator(trainingData, testData=testData, **self.evaluatorParams)

    def _createResultData(self, trainedModels, evalDataList, testIndicesList, predictedVarNames) -> VectorRegressionModelCrossValidationData:
        return VectorRegressionModelCrossValidationData(trainedModels, evalDataList, predictedVarNames, testIndicesList)


class VectorClassificationModelCrossValidationData(VectorModelCrossValidationData[VectorClassificationModel, VectorClassificationModelEvaluationData, ClassificationEvalStats, ClassificationEvalStatsCollection]):
    def _createEvalStatsCollection(self, l: List[ClassificationEvalStats]) -> ClassificationEvalStatsCollection:
        return ClassificationEvalStatsCollection(l)


class VectorClassificationModelCrossValidator(VectorModelCrossValidator[VectorClassificationModelCrossValidationData]):
    def _createModelEvaluator(self, trainingData: InputOutputData, testData: InputOutputData):
        return VectorClassificationModelEvaluator(trainingData, testData=testData, **self.evaluatorParams)

    def _createResultData(self, trainedModels, evalDataList, testIndicesList, predictedVarNames) -> VectorClassificationModelCrossValidationData:
        return VectorClassificationModelCrossValidationData(trainedModels, evalDataList, predictedVarNames, testIndicesList)
