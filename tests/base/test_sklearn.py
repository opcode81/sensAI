import sensai


def test_RandomForestClassifier(irisClassificationTestCase):
    model = sensai.sklearn.classification.SkLearnRandomForestVectorClassificationModel()
    irisClassificationTestCase.testMinAccuracy(model, 0.9)


def test_MLP(irisClassificationTestCase):
    mlpBFGS = sensai.sklearn.classification.SkLearnMLPVectorClassificationModel(solver="lbfgs").with_name("skMLP-lbfgs")
    irisClassificationTestCase.testMinAccuracy(mlpBFGS, 0.9)
    mlpAdam = sensai.sklearn.classification.SkLearnMLPVectorClassificationModel(solver="adam").with_name("skMLP-adam")
    irisClassificationTestCase.testMinAccuracy(mlpAdam, 0.9)
