import logging
import os
import sys

import pandas as pd
import pytest
import sklearn.datasets

from sensai import InputOutputData, VectorClassificationModel, VectorRegressionModel
from sensai.evaluation import VectorClassificationModelEvaluator, VectorRegressionModelEvaluator, VectorRegressionModelEvaluatorParams

sys.path.append(os.path.abspath("."))
from config import topLevelDirectory

log = logging.getLogger(__name__)


RESOURCE_DIR = os.path.join(topLevelDirectory, "tests", "resources")


@pytest.fixture(scope="session")
def testResources():
    return RESOURCE_DIR


class IrisDataSet:
    _iod = None

    @classmethod
    def getInputOutputData(cls):
        if cls._iod is None:
            d = sklearn.datasets.load_iris()
            inputs = pd.DataFrame(d["data"], columns=d["feature_names"])
            targetNames = d["target_names"]
            outputs = pd.DataFrame({"class": [targetNames[x] for x in d["target"]]})
            cls._iod = InputOutputData(inputs, outputs)
        return cls._iod


class ClassificationTestCase:
    def __init__(self, data: InputOutputData):
        self.data = data

    def testMinAccuracy(self, model: VectorClassificationModel, minAccuracy: float, fit=True):
        ev = VectorClassificationModelEvaluator(self.data, testFraction=0.2)
        if fit:
            ev.fitModel(model)
        resultData = ev.evalModel(model)
        stats = resultData.getEvalStats()
        #stats.plotConfusionMatrix().savefig("cmat.png")
        log.info(f"Results for {model.getName()}: {stats}")
        assert stats.getAccuracy() >= minAccuracy


@pytest.fixture(scope="session")
def irisDataSet():
    return IrisDataSet()


@pytest.fixture(scope="session")
def irisClassificationTestCase(irisDataSet):
    return ClassificationTestCase(irisDataSet.getInputOutputData())


class RegressionTestCase:
    def __init__(self, data: InputOutputData):
        self.data = data
        self.evaluatorParams = VectorRegressionModelEvaluatorParams(fractionalSplitTestFraction=0.2, fractionalSplitShuffle=True,
            fractionalSplitRandomSeed=42)

    def testMinR2(self, model: VectorRegressionModel, minR2: float, fit=True):
        ev = VectorRegressionModelEvaluator(self.data, params=self.evaluatorParams)
        if fit:
            ev.fitModel(model)
        resultData = ev.evalModel(model)
        stats = resultData.getEvalStats()

        #stats.plotScatterGroundTruthPredictions()
        #from matplotlib import pyplot as plt; plt.show()
        #resultDataTrain = ev.evalModel(model, onTrainingData=True); log.info(f"on train: {resultDataTrain.getEvalStats()}")

        log.info(f"Results for {model.getName()}: {stats}")
        assert stats.getR2() >= minR2


class DiabetesDataSet:
    """
    Classic diabetes data set (downloaded from https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt)
    """
    _iod = None

    @classmethod
    def getInputOutputData(cls):
        if cls._iod is None:
            df = pd.read_csv(os.path.join(RESOURCE_DIR, "diabetes.tab.txt"), sep="\t")
            return InputOutputData.fromDataFrame(df, "Y")
        return cls._iod


@pytest.fixture(scope="session")
def diabetesDataSet():
    return DiabetesDataSet()


@pytest.fixture(scope="session")
def diabetesRegressionTestCase(diabetesDataSet):
    return RegressionTestCase(diabetesDataSet.getInputOutputData())
