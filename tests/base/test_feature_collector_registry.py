from enum import Enum

from sensai.data_transformation import DFTNormalisation, SkLearnTransformerFactoryFactory
from sensai.featuregen import FeatureGeneratorRegistry, FeatureGeneratorTakeColumns


def test_feature_collector_with_registry(irisClassificationTestCase):
    class FeatureName(Enum):
        ALL = "all"

    registry = FeatureGeneratorRegistry()
    registry.register_factory(FeatureName.ALL, lambda: FeatureGeneratorTakeColumns(
        normalisation_rule_template=DFTNormalisation.RuleTemplate(independent_columns=True)))

    fc = registry.collect_features(FeatureName.ALL)

    fgen = fc.get_multi_feature_generator()
    features_df = fgen.fit_generate(irisClassificationTestCase.data.inputs)

    dft_normalisation = fc.create_dft_normalisation(default_transformer_factory=SkLearnTransformerFactoryFactory.MaxAbsScaler())
    normalised_features_df = dft_normalisation.fit_apply(features_df)

    max_values = normalised_features_df.max(axis=0)
    assert all(max_values == 1.0)  # test correctness of independent_columns=True
