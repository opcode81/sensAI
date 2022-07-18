import logging
import random

import numpy as np
import pandas as pd
import pytest

from sensai import InputOutputData
from sensai.data_transformation import DFTNormalisation, DFTFillNA, DataFrameTransformer
from sensai.data_transformation.sklearn_transformer import SkLearnTransformerFactoryFactory
from sensai.evaluation import VectorClassificationModelEvaluator
from sensai.featuregen import FeatureGeneratorFlattenColumns, FeatureGeneratorTakeColumns, flattenedFeatureGenerator, \
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
    fgen1 = flattenedFeatureGenerator(FeatureGeneratorTakeColumns("a"))
    assert fgen1.generate(inputDf).equals(pd.DataFrame({"a_0": np.array([1]), "a_1": np.array([2])}))


class TestFgen(FeatureGenerator):
    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame = None, ctx=None):
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

    @pytest.mark.parametrize("testfgen", [TestFgen(), TestDFT().toFeatureGenerator()])
    def test_basicProperties(self, testfgen):
        assert not testfgen.isFitted()
        assert testfgen.getGeneratedColumnNames() is None
        testfgen.fit(self.testdf)
        assert testfgen.isFitted()
        testfgen.generate(self.testdf)
        assert set(testfgen.getGeneratedColumnNames()) == {"foo", "bar"}
    
    @pytest.mark.parametrize("fgen", [TestFgen(), TestDFT().toFeatureGenerator(), RuleBasedTestFgen(), MultiFeatureGenerator(TestFgen()),
        ChainedFeatureGenerator(TestFgen())])
    def test_Naming(self, fgen):
        assert isinstance(fgen.getName(), str)
        fgen.setName("bar")
        assert fgen.getName() == "bar"

    def test_ruleBasedAlwaysFitted(self):
        assert RuleBasedTestFgen().isFitted()

    def test_emptyChainRaisesError(self):
        with pytest.raises(ValueError):
            ChainedFeatureGenerator()

    def test_combinationFittedIffEachMemberFitted(self):
        # if one of the fgens is not fitted, the combination is not fitted either
        multifgen = MultiFeatureGenerator(TestFgen(), RuleBasedTestFgen())
        chainfgen = ChainedFeatureGenerator(TestFgen(), RuleBasedTestFgen())
        assert chainfgen.featureGenerators[1].isFitted() and multifgen.featureGenerators[1].isFitted()
        assert not multifgen.isFitted() and not chainfgen.isFitted()
        chainfgen.fit(self.testdf)
        multifgen.fit(self.testdf)
        assert multifgen.isFitted() and chainfgen.isFitted()
        assert chainfgen.featureGenerators[0].isFitted() and multifgen.featureGenerators[0].isFitted()

        # if all fgens are fitted, the combination is also fitted, even if fit was not called
        multifgen = MultiFeatureGenerator(RuleBasedTestFgen(), RuleBasedTestFgen())
        chainfgen = ChainedFeatureGenerator(RuleBasedTestFgen(), RuleBasedTestFgen())
        assert multifgen.isFitted() and chainfgen.isFitted()


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
        fgs = [FeatureGeneratorTakeColumns(normalisationRuleTemplate=DFTNormalisation.RuleTemplate(independentColumns=True))]
        if useFGNA:
            fgs.append(FeatureGeneratorNAMarker(inputs.columns))
        fCollector = FeatureCollector(*fgs)
        model = SkLearnMLPVectorClassificationModel() \
            .withFeatureCollector(fCollector) \
            .withInputTransformers(
                DFTNormalisation(fCollector.getNormalisationRules(), defaultTransformerFactory=SkLearnTransformerFactoryFactory.StandardScaler()),
                DFTFillNA(-3))
        # NOTE: using -3 instead of 0 to fill N/A values in order to force the model to learn the purpose of the N/A markers,
        # because 0 values are actually a reasonable fallback (which happens to work) when using StandardScaler
        # NOTE: it is important to apply DFTNormalisation before DFTFillNA, because DFTNormalisation would learn using the filled values otherwise

        ev = VectorClassificationModelEvaluator(iodata, testFraction=0.2)
        ev.fitModel(model)
        result = ev.evalModel(model)
        accuracy = result.getEvalStats().getAccuracy()
        log.info(f"Accuracy (for useFGNA={useFGNA}) = {accuracy}")
        if useFGNA:
            assert accuracy > 0.85
        else:
            assert accuracy < 0.85