import logging
import random

import numpy as np
import pandas as pd
import pytest

from sensai import InputOutputData
from sensai.data_transformation import DFTNormalisation, DFTFillNA, DataFrameTransformer
from sensai.data_transformation.sklearn_transformer import SkLearnTransformerFactoryFactory
from sensai.evaluation import VectorClassificationModelEvaluator, VectorClassificationModelEvaluatorParams
from sensai.featuregen import FeatureGeneratorFlattenColumns, FeatureGeneratorTakeColumns, flattened_feature_generator, \
    FeatureGenerator, RuleBasedFeatureGenerator, MultiFeatureGenerator, ChainedFeatureGenerator, FeatureGeneratorNAMarker, FeatureCollector
from sensai.sklearn.sklearn_classification import SkLearnMLPVectorClassificationModel


log = logging.getLogger(__name__)


def test_take_columns():
    inputDf = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    fgen1 = FeatureGeneratorTakeColumns("a")
    fgen2 = FeatureGeneratorTakeColumns()
    fgen3 = FeatureGeneratorTakeColumns(["a", "b"])
    fgen4 = FeatureGeneratorTakeColumns(["d"])

    assert fgen1.generate(inputDf).equals(inputDf[["a"]])
    assert fgen2.generate(inputDf).equals(inputDf)
    assert fgen3.generate(inputDf).equals(inputDf[["a", "b"]])
    with pytest.raises(Exception):
        fgen4.generate(inputDf)

def test_flatten_columns():
    inputDf = pd.DataFrame({"a": [np.array([1, 2])], "b": [np.array([5, 6])]})
    fgen1 = FeatureGeneratorFlattenColumns("a")
    fgen2 = FeatureGeneratorFlattenColumns()
    fgen3 = FeatureGeneratorFlattenColumns(["a"])
    fgen4 = FeatureGeneratorFlattenColumns("c")
    assert fgen1.generate(inputDf).equals(pd.DataFrame({"a_0": np.array([1]), "a_1": np.array([2])}))
    assert fgen2.generate(inputDf).equals(pd.DataFrame({"a_0": np.array([1]), "a_1": np.array([2]), "b_0": np.array([5]), "b_1": np.array([6])}))
    assert fgen3.generate(inputDf).equals(pd.DataFrame({"a_0": np.array([1]), "a_1": np.array([2])}))
    with pytest.raises(Exception):
        assert fgen4.generate(inputDf)


def test_getFlattenedFeatureGenerator():
    inputDf = pd.DataFrame({"a": [np.array([1, 2])], "b": [np.array([5, 6])]})
    fgen1 = flattened_feature_generator(FeatureGeneratorTakeColumns("a"))
    assert fgen1.generate(inputDf).equals(pd.DataFrame({"a_0": np.array([1]), "a_1": np.array([2])}))


class TestFgen(FeatureGenerator):
    def _fit(self, x: pd.DataFrame, y: pd.DataFrame = None, ctx=None):
        pass

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        return df


class TestDFT(DataFrameTransformer):
    def _fit(self, df: pd.DataFrame):
        pass

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class RuleBasedTestFgen(RuleBasedFeatureGenerator):
    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        return df


class TestFgenBasics:
    testdf = pd.DataFrame({"foo": [1, 2], "bar": [1, 2]})

    @pytest.mark.parametrize("testfgen", [TestFgen(), TestDFT().to_feature_generator()])
    def test_basicProperties(self, testfgen):
        assert not testfgen.is_fitted()
        assert testfgen.get_generated_column_names() is None
        testfgen.fit(self.testdf)
        assert testfgen.is_fitted()
        testfgen.generate(self.testdf)
        assert set(testfgen.get_generated_column_names()) == {"foo", "bar"}
    
    @pytest.mark.parametrize("fgen", [TestFgen(), TestDFT().to_feature_generator(), RuleBasedTestFgen(), MultiFeatureGenerator(TestFgen()),
        ChainedFeatureGenerator(TestFgen())])
    def test_Naming(self, fgen):
        assert isinstance(fgen.get_name(), str)
        fgen.set_name("bar")
        assert fgen.get_name() == "bar"

    def test_ruleBasedAlwaysFitted(self):
        assert RuleBasedTestFgen().is_fitted()

    def test_emptyChainInvalid(self):
        with pytest.raises(ValueError):
            ChainedFeatureGenerator()

    def test_combinationFittedIffEachMemberFitted(self):
        # if one of the fgens is not fitted, the combination is not fitted either
        multifgen = MultiFeatureGenerator(TestFgen(), RuleBasedTestFgen())
        chainfgen = ChainedFeatureGenerator(TestFgen(), RuleBasedTestFgen())
        assert chainfgen.featureGenerators[1].is_fitted() and multifgen.featureGenerators[1].is_fitted()
        assert not multifgen.is_fitted() and not chainfgen.is_fitted()
        chainfgen.fit(self.testdf)
        multifgen.fit(self.testdf)
        assert multifgen.is_fitted() and chainfgen.is_fitted()
        assert chainfgen.featureGenerators[0].is_fitted() and multifgen.featureGenerators[0].is_fitted()

        # if all fgens are fitted, the combination is also fitted, even if fit was not called
        multifgen = MultiFeatureGenerator(RuleBasedTestFgen(), RuleBasedTestFgen())
        chainfgen = ChainedFeatureGenerator(RuleBasedTestFgen(), RuleBasedTestFgen())
        assert multifgen.is_fitted() and chainfgen.is_fitted()


def test_FeatureGeneratorNAMarker(irisClassificationTestCase):
    """
    Integration test for handling of N/A values via marker features (using FeatureGeneratorNAMarker) in the context of models
    that do not support N/A values, replacing them with a different value (using FillNA)
    """
    iodata = irisClassificationTestCase.data

    # create some random N/A values in the data set
    inputs = iodata.inputs.copy()
    rand = random.Random(42)
    fullIndices = list(range(len(inputs)))
    for col in inputs.columns:
        indices = rand.sample(fullIndices, 20)
        inputs[col].iloc[indices] = np.nan
    iodata = InputOutputData(inputs, iodata.outputs)

    for useFGNA in (True, False):
        fgs = [FeatureGeneratorTakeColumns(normalisation_rule_template=DFTNormalisation.RuleTemplate(independent_columns=True))]
        if useFGNA:
            fgs.append(FeatureGeneratorNAMarker(inputs.columns))
        fCollector = FeatureCollector(*fgs)
        model = SkLearnMLPVectorClassificationModel() \
            .with_feature_collector(fCollector) \
            .with_input_transformers(
                DFTNormalisation(fCollector.get_normalisation_rules(), default_transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler()),
                DFTFillNA(-3))
        # NOTE: using -3 instead of 0 to fill N/A values in order to force the model to learn the purpose of the N/A markers,
        # because 0 values are actually a reasonable fallback (which happens to work) when using StandardScaler
        # NOTE: it is important to apply DFTNormalisation before DFTFillNA, because DFTNormalisation would learn using the filled values otherwise

        ev = VectorClassificationModelEvaluator(iodata, params=VectorClassificationModelEvaluatorParams(fractional_split_test_fraction=0.2))
        ev.fit_model(model)
        result = ev.eval_model(model)
        accuracy = result.get_eval_stats().get_accuracy()
        log.info(f"Accuracy (for useFGNA={useFGNA}) = {accuracy}")
        if useFGNA:
            assert accuracy > 0.85
        else:
            assert accuracy < 0.85