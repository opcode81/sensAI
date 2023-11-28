import os
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

from sensai import InputOutputData, VectorClassificationModel, VectorRegressionModel
from sensai.evaluation import VectorClassificationModelEvaluator, VectorRegressionModelEvaluatorParams, VectorRegressionModelEvaluator, \
    VectorClassificationModelEvaluatorParams
from sensai.util import logging

log = logging.getLogger(__name__)
RESOURCE_PATH = Path(__file__).resolve().parent / "resources"
RESOURCE_DIR = str(RESOURCE_PATH)


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
        ev = VectorClassificationModelEvaluator(self.data,
            params=VectorClassificationModelEvaluatorParams(fractional_split_test_fraction=0.2, fractional_split_random_seed=42,
                fractional_split_shuffle=True))
        if fit:
            ev.fit_model(model)
        resultData = ev.eval_model(model)
        stats = resultData.get_eval_stats()
        #stats.plotConfusionMatrix().savefig("cmat.png")
        log.info(f"Results for {model.get_name()}: {stats}")
        assert stats.get_accuracy() >= minAccuracy


class RegressionTestCase:
    def __init__(self, data: InputOutputData):
        self.data = data
        self.evaluatorParams = VectorRegressionModelEvaluatorParams(fractional_split_test_fraction=0.2, fractional_split_shuffle=True,
            fractional_split_random_seed=42)

    def createEvaluator(self) -> VectorRegressionModelEvaluator:
        return VectorRegressionModelEvaluator(self.data, params=self.evaluatorParams)

    def testMinR2(self, model: VectorRegressionModel, minR2: float, fit=True):
        ev = self.createEvaluator()
        if fit:
            ev.fit_model(model)
        resultData = ev.eval_model(model)
        stats = resultData.get_eval_stats()

        #stats.plotScatterGroundTruthPredictions()
        #from matplotlib import pyplot as plt; plt.show()
        #resultDataTrain = ev.evalModel(model, onTrainingData=True); log.info(f"on train: {resultDataTrain.getEvalStats()}")

        log.info(f"Results for {model.get_name()}: {stats}")
        assert stats.compute_r2() >= minR2


class DiabetesDataSet:
    """
    Classic diabetes data set (downloaded from https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt)
    """
    _iod = None
    categorical_features = ["SEX"]
    numeric_features = ["AGE", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6"]

    @classmethod
    def getInputOutputData(cls):
        if cls._iod is None:
            df = pd.read_csv(os.path.join(RESOURCE_DIR, "diabetes.tab.txt"), sep="\t")
            return InputOutputData.from_data_frame(df, "Y")
        return cls._iod
