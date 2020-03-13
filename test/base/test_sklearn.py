import sensai
from ..classification import IrisClassificationTestCase


def test_RandomForestClassifier():
    model = sensai.sklearn.classification.SkLearnRandomForestVectorClassificationModel()
    IrisClassificationTestCase().testMinAccuracy(model, 0.9)


def test_MLP():
    testCase = IrisClassificationTestCase()
    mlpBFGS = sensai.sklearn.classification.SkLearnMLPVectorClassificationModel(solver="lbfgs").withName("skMLP-lbfgs")
    testCase.testMinAccuracy(mlpBFGS, 0.9)
    mlpAdam = sensai.sklearn.classification.SkLearnMLPVectorClassificationModel(solver="adam").withName("skMLP-adam")
    testCase.testMinAccuracy(mlpAdam, 0.9)
