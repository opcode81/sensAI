import sys

from sensai.data_transformation import DFTNormalisation, DFTOneHotEncoder, SkLearnTransformerFactoryFactory
from sensai.featuregen import FeatureCollector, FeatureGeneratorTakeColumns
from sensai.sklearn.sklearn_regression import SkLearnLinearRegressionVectorRegressionModel, SkLearnRandomForestVectorRegressionModel, \
    SkLearnMultiLayerPerceptronVectorRegressionModel
from sensai.util import logging
from sensai.util.pickle import dump_pickle
from tests.model_test_case import DiabetesDataSet, RegressionTestCase, RESOURCE_PATH


def create_regression_models_for_backward_compatibility_test(version):
    """
    :param version: version with which the files are created
    """
    dataset = DiabetesDataSet()

    fc = FeatureCollector(
        FeatureGeneratorTakeColumns(dataset.categorical_features, categorical_feature_names=dataset.categorical_features,
            normalisation_rule_template=DFTNormalisation.RuleTemplate(unsupported=True)),
        FeatureGeneratorTakeColumns(dataset.numeric_features,
            normalisation_rule_template=DFTNormalisation.RuleTemplate(independent_columns=True)))

    modelLinear = SkLearnLinearRegressionVectorRegressionModel() \
        .with_feature_collector(fc) \
        .with_feature_transformers(
            DFTOneHotEncoder(fc.get_categorical_feature_name_regex()),
            DFTNormalisation(fc.get_normalisation_rules(), default_transformer_factory=SkLearnTransformerFactoryFactory.RobustScaler())) \
        .with_name("Linear")

    modelRF = SkLearnRandomForestVectorRegressionModel(n_estimators=10, min_samples_leaf=10) \
        .with_feature_collector(fc) \
        .with_feature_transformers(DFTOneHotEncoder(fc.get_categorical_feature_name_regex())) \
        .with_name("RandomForest")

    modelMLP = SkLearnMultiLayerPerceptronVectorRegressionModel(hidden_layer_sizes=(20,20), solver="adam", max_iter=1000, batch_size=32, early_stopping=True) \
        .with_feature_collector(fc) \
        .with_feature_transformers(
            DFTOneHotEncoder(fc.get_categorical_feature_name_regex()),
            DFTNormalisation(fc.get_normalisation_rules(), default_transformer_factory=SkLearnTransformerFactoryFactory.RobustScaler())) \
        .with_name("SkLearnMLP")

    testCase = RegressionTestCase(dataset.getInputOutputData())
    ev = testCase.createEvaluator()
    for model in [modelLinear, modelRF, modelMLP]:
        ev.fit_model(model)
        eval_data = ev.eval_model(model)
        eval_stats = eval_data.get_eval_stats()
        print(eval_stats)
        r2 = eval_stats.compute_r2()
        persisted_data = {"R2": r2, "model": model}
        dump_pickle(persisted_data, RESOURCE_PATH / "backward_compatibility" / f"regression_model_{model.get_name()}.{version}.pickle",
            protocol=4)


if __name__ == '__main__':
    logging.configure()
    sys.path.append("../..")
    create_regression_models_for_backward_compatibility_test("v0.2.0")
