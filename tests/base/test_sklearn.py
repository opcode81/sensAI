import pytest
from sklearn.preprocessing import StandardScaler

import sensai
from sensai.data_transformation import DFTSkLearnTransformer
from sensai.sklearn.sklearn_regression import SkLearnRandomForestVectorRegressionModel, SkLearnLinearRegressionVectorRegressionModel, \
    SkLearnLinearRidgeRegressionVectorRegressionModel, SkLearnLinearLassoRegressionVectorRegressionModel, SkLearnMultiLayerPerceptronVectorRegressionModel, \
    SkLearnSVRVectorRegressionModel, SkLearnDummyVectorRegressionModel, SkLearnDecisionTreeVectorRegressionModel


def test_classifier_RandomForestClassifier(irisClassificationTestCase):
    model = sensai.sklearn.classification.SkLearnRandomForestVectorClassificationModel()
    irisClassificationTestCase.testMinAccuracy(model, 0.9)


def test_classifier_MLP(irisClassificationTestCase):
    mlpBFGS = sensai.sklearn.classification.SkLearnMLPVectorClassificationModel(solver="lbfgs").withName("skMLP-lbfgs")
    irisClassificationTestCase.testMinAccuracy(mlpBFGS, 0.9)
    mlpAdam = sensai.sklearn.classification.SkLearnMLPVectorClassificationModel(solver="adam").withName("skMLP-adam")
    irisClassificationTestCase.testMinAccuracy(mlpAdam, 0.9)


@pytest.mark.parametrize("min_r2,model_factory", [
    (0.41, lambda: SkLearnRandomForestVectorRegressionModel(n_estimators=100, min_samples_leaf=5).withName("RF")),
    (0.38, lambda: SkLearnLinearRegressionVectorRegressionModel().withFeatureTransformers(DFTSkLearnTransformer(StandardScaler())).withName("Linear")),
    (0.38, lambda: SkLearnLinearRidgeRegressionVectorRegressionModel().withFeatureTransformers(DFTSkLearnTransformer(StandardScaler())).withName("Ridge")),
    (0.38, lambda: SkLearnLinearLassoRegressionVectorRegressionModel().withFeatureTransformers(DFTSkLearnTransformer(StandardScaler())).withName("Lasso")),
    (0.49, lambda: SkLearnMultiLayerPerceptronVectorRegressionModel(hidden_layer_sizes=(20,20), solver="adam", max_iter=1000, batch_size=32, early_stopping=True).withFeatureTransformers(DFTSkLearnTransformer(StandardScaler())).withName("MLP-adam")),
    (0.18, lambda: SkLearnSVRVectorRegressionModel().withFeatureTransformers(DFTSkLearnTransformer(StandardScaler())).withName("SVR")),
    (-0.05, lambda: SkLearnDummyVectorRegressionModel(strategy="mean").withName("Mean")),
    (0.28, lambda: SkLearnDecisionTreeVectorRegressionModel(min_samples_leaf=20).withName("DecisionTree")),
])
def test_SkLearnVectorRegressionModels(diabetesRegressionTestCase, model_factory, min_r2):
    model = model_factory()
    diabetesRegressionTestCase.testMinR2(model, min_r2)
