import sys

from sensai.data_transformation import DFTNormalisation, DFTOneHotEncoder, SkLearnTransformerFactoryFactory
from sensai.featuregen import FeatureCollector, FeatureGeneratorTakeColumns
from sensai.sklearn.sklearn_regression import SkLearnLinearRegressionVectorRegressionModel, SkLearnRandomForestVectorRegressionModel, \
    SkLearnMultiLayerPerceptronVectorRegressionModel
from sensai.util import logging
from sensai.util.pickle import dumpPickle
from tests.model_test_case import DiabetesDataSet, RegressionTestCase, RESOURCE_PATH


def create_regression_models_for_backward_compatibility_test(version):
    """
    :param version: version with which the files are created
    """
    dataset = DiabetesDataSet()

    fc = FeatureCollector(
        FeatureGeneratorTakeColumns(dataset.categorical_features, categoricalFeatureNames=dataset.categorical_features,
            normalisationRuleTemplate=DFTNormalisation.RuleTemplate(unsupported=True)),
        FeatureGeneratorTakeColumns(dataset.numeric_features,
            normalisationRuleTemplate=DFTNormalisation.RuleTemplate(independentColumns=True)))

    modelLinear = SkLearnLinearRegressionVectorRegressionModel() \
        .withFeatureCollector(fc) \
        .withFeatureTransformers(
            DFTOneHotEncoder(fc.getCategoricalFeatureNameRegex()),
            DFTNormalisation(fc.getNormalisationRules(), defaultTransformerFactory=SkLearnTransformerFactoryFactory.RobustScaler())) \
        .withName("Linear")

    modelRF = SkLearnRandomForestVectorRegressionModel(n_estimators=10, min_samples_leaf=10) \
        .withFeatureCollector(fc) \
        .withFeatureTransformers(DFTOneHotEncoder(fc.getCategoricalFeatureNameRegex())) \
        .withName("RandomForest")

    modelMLP = SkLearnMultiLayerPerceptronVectorRegressionModel(hidden_layer_sizes=(20,20), solver="adam", max_iter=1000, batch_size=32, early_stopping=True) \
        .withFeatureCollector(fc) \
        .withFeatureTransformers(
            DFTOneHotEncoder(fc.getCategoricalFeatureNameRegex()),
            DFTNormalisation(fc.getNormalisationRules(), defaultTransformerFactory=SkLearnTransformerFactoryFactory.RobustScaler())) \
        .withName("SkLearnMLP")

    testCase = RegressionTestCase(dataset.getInputOutputData())
    ev = testCase.createEvaluator()
    for model in [modelLinear, modelRF, modelMLP]:
        ev.fitModel(model)
        eval_data = ev.evalModel(model)
        eval_stats = eval_data.getEvalStats()
        print(eval_stats)
        r2 = eval_stats.getR2()
        persisted_data = {"R2": r2, "model": model}
        dumpPickle(persisted_data, RESOURCE_PATH / "backward_compatibility" / f"regression_model_{model.getName()}.{version}.pickle")


if __name__ == '__main__':
    logging.configureLogging()
    sys.path.append("../..")
    create_regression_models_for_backward_compatibility_test("v0.2.0")
