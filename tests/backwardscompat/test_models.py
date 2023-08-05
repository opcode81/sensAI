import os

import sensai
from sensai import VectorModel
from sensai.data_transformation import DFTNormalisation, SkLearnTransformerFactoryFactory, DFTOneHotEncoder
from sensai.featuregen import FeatureGeneratorTakeColumns, FeatureCollector
from sensai.sklearn.sklearn_regression import SkLearnLinearRegressionVectorRegressionModel, SkLearnRandomForestVectorRegressionModel, \
    SkLearnMultiLayerPerceptronVectorRegressionModel
from tests.conftest import RegressionTestCase


def test_modelCanBeLoaded(testResources, irisClassificationTestCase):
    # The model file was generated with tests/frameworks/torch/test_torch.test_MLPClassifier at commit f93c6b11d
    modelPath = os.path.join(testResources, "torch_mlp.pickle")
    model = VectorModel.load(modelPath)
    assert isinstance(model, sensai.torch.models.MultiLayerPerceptronVectorClassificationModel)
    irisClassificationTestCase.testMinAccuracy(model, 0.8, fit=False)


# TODO
def createRegressionModelsForBackwardsCompatibilityTest(testCase: RegressionTestCase):
    fc = FeatureCollector(FeatureGeneratorTakeColumns(categoricalFeatureNames=["SEX"],
        normalisationRuleTemplate=DFTNormalisation.RuleTemplate(independentColumns=False)))

    modelLinear = SkLearnLinearRegressionVectorRegressionModel() \
        .withFeatureCollector(fc) \
        .withFeatureTransformers(
            DFTOneHotEncoder(fc.getCategoricalFeatureNameRegex()))
            #DFTNormalisation(fc.getNormalisationRules(), defaultTransformerFactory=SkLearnTransformerFactoryFactory.RobustScaler()))

    modelRF = SkLearnRandomForestVectorRegressionModel() \
        .withFeatureCollector(fc) \
        .withFeatureTransformers(DFTOneHotEncoder(fc.getCategoricalFeatureNameRegex()))

    modelMLP = SkLearnMultiLayerPerceptronVectorRegressionModel(hidden_layer_sizes=(10, 10), solver="lbfgs") \
        .withFeatureCollector(fc) \
        .withFeatureTransformers(
            DFTOneHotEncoder(fc.getCategoricalFeatureNameRegex()),
            DFTNormalisation(fc.getNormalisationRules(), defaultTransformerFactory=SkLearnTransformerFactoryFactory.RobustScaler()))

    return modelMLP


# TODO
def todo_test_backward_compatibility_v020(diabetesRegressionTestCase):
    model = createRegressionModelsForBackwardsCompatibilityTest(diabetesRegressionTestCase)
    diabetesRegressionTestCase.testMinR2(model, 0.5, fit=True)