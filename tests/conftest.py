import logging
import os
import sys

import pandas as pd
import pytest
import sklearn.datasets

from sensai import InputOutputData, VectorClassificationModel
from sensai.evaluation import VectorClassificationModelEvaluator, VectorClassificationModelEvaluatorParams

sys.path.append(os.path.abspath("."))
from config import topLevelDirectory

log = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def testResources():
    return os.path.join(topLevelDirectory, "tests", "resources")


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
