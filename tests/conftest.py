import logging
import os
import sys

import pandas as pd
import pytest
from sklearn.datasets import load_iris

from sensai import InputOutputData, VectorClassificationModel, VectorRegressionModel
from sensai.evaluation import VectorClassificationModelEvaluator, VectorRegressionModelEvaluator, VectorRegressionModelEvaluatorParams, \
    VectorClassificationModelEvaluatorParams

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
            d = load_iris()
            inputs = pd.DataFrame(d["data"], columns=d["feature_names"])
            targetNames = d["target_names"]
            outputs = pd.DataFrame({"class": [targetNames[x] for x in d["target"]]})
            cls._iod = InputOutputData(inputs, outputs)
        return cls._iod


class ClassificationTestCase:
    def __init__(self, data: InputOutputData):
        self.data = data

    def testMinAccuracy(self, model: VectorClassificationModel, minAccuracy: float, fit=True):
        params = VectorClassificationModelEvaluatorParams(fractional_split_test_fraction=0.2)
        ev = VectorClassificationModelEvaluator(self.data, params=params)
        if fit:
            ev.fit_model(model)
        resultData = ev.eval_model(model)
        stats = resultData.get_eval_stats()
        #stats.plotConfusionMatrix().savefig("cmat.png")
        log.info(f"Results for {model.get_name()}: {stats}")
        assert stats.get_accuracy() >= minAccuracy


@pytest.fixture(scope="session")
def irisDataSet():
    return IrisDataSet()


@pytest.fixture(scope="session")
def irisClassificationTestCase(irisDataSet):
    return ClassificationTestCase(irisDataSet.getInputOutputData())


class RegressionTestCase:
    def __init__(self, data: InputOutputData):
        self.data = data
        self.evaluatorParams = VectorRegressionModelEvaluatorParams(fractional_split_test_fraction=0.2, fractional_split_shuffle=True,
            fractional_split_random_seed=42)

    def testMinR2(self, model: VectorRegressionModel, minR2: float, fit=True):
        ev = VectorRegressionModelEvaluator(self.data, params=self.evaluatorParams)
        if fit:
            ev.fit_model(model)
        resultData = ev.eval_model(model)
        stats = resultData.get_eval_stats()

        #stats.plotScatterGroundTruthPredictions()
        #from matplotlib import pyplot as plt; plt.show()
        #resultDataTrain = ev.evalModel(model, onTrainingData=True); log.info(f"on train: {resultDataTrain.getEvalStats()}")

        log.info(f"Results for {model.get_name()}: {stats}")
        assert stats.get_r2() >= minR2


class DiabetesDataSet:
    """
    Classic diabetes data set (downloaded from https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt)
    """
    _iod = None

    @classmethod
    def getInputOutputData(cls):
        if cls._iod is None:
            df = pd.read_csv(os.path.join(RESOURCE_DIR, "diabetes.tab.txt"), sep="\t")
            return InputOutputData.from_data_frame(df, "Y")
        return cls._iod


@pytest.fixture(scope="session")
def diabetesDataSet():
    return DiabetesDataSet()


@pytest.fixture(scope="session")
def diabetesRegressionTestCase(diabetesDataSet):
    return RegressionTestCase(diabetesDataSet.getInputOutputData())
