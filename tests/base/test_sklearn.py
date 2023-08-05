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
    mlpBFGS = sensai.sklearn.classification.SkLearnMLPVectorClassificationModel(solver="lbfgs").with_name("skMLP-lbfgs")
    irisClassificationTestCase.testMinAccuracy(mlpBFGS, 0.9)
    mlpAdam = sensai.sklearn.classification.SkLearnMLPVectorClassificationModel(solver="adam").with_name("skMLP-adam")
    irisClassificationTestCase.testMinAccuracy(mlpAdam, 0.9)


@pytest.mark.parametrize("min_r2,model_factory", [
    (0.41, lambda: SkLearnRandomForestVectorRegressionModel(n_estimators=100, min_samples_leaf=5).with_name("RF")),
    (0.38, lambda: SkLearnLinearRegressionVectorRegressionModel().with_feature_transformers(DFTSkLearnTransformer(StandardScaler())).with_name("Linear")),
    (0.38, lambda: SkLearnLinearRidgeRegressionVectorRegressionModel().with_feature_transformers(DFTSkLearnTransformer(StandardScaler())).with_name("Ridge")),
    (0.38, lambda: SkLearnLinearLassoRegressionVectorRegressionModel().with_feature_transformers(DFTSkLearnTransformer(StandardScaler())).with_name("Lasso")),
    (0.49, lambda: SkLearnMultiLayerPerceptronVectorRegressionModel(hidden_layer_sizes=(20,20), solver="adam", max_iter=1000, batch_size=32, early_stopping=True).with_feature_transformers(DFTSkLearnTransformer(StandardScaler())).with_name("MLP-adam")),
    (0.18, lambda: SkLearnSVRVectorRegressionModel().with_feature_transformers(DFTSkLearnTransformer(StandardScaler())).with_name("SVR")),
    (-0.05, lambda: SkLearnDummyVectorRegressionModel(strategy="mean").with_name("Mean")),
    (0.28, lambda: SkLearnDecisionTreeVectorRegressionModel(min_samples_leaf=20).with_name("DecisionTree")),
])
def test_SkLearnVectorRegressionModels(diabetesRegressionTestCase, model_factory, min_r2):
    model = model_factory()
    diabetesRegressionTestCase.testMinR2(model, min_r2)
