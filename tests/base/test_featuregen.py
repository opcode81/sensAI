import numpy as np
import pandas as pd
import pytest

from sensai.featuregen import FeatureGeneratorFlattenColumns, FeatureGeneratorTakeColumns, flattenedFeatureGenerator, \
    FeatureGenerator, RuleBasedFeatureGenerator, MultiFeatureGenerator, ChainedFeatureGenerator


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
    assert fgen1.generate(inputDf).equals(pd.DataFrame({"a_0": [1], "a_1": [2]}))
    assert fgen2.generate(inputDf).equals(pd.DataFrame({"a_0": [1], "a_1": [2], "b_0": [5], "b_1": [6]}))
    assert fgen3.generate(inputDf).equals(pd.DataFrame({"a_0": [1], "a_1": [2]}))
    with pytest.raises(Exception):
        assert fgen4.generate(inputDf)


def test_getFlattenedFeatureGenerator():
    inputDf = pd.DataFrame({"a": [np.array([1, 2])], "b": [np.array([5, 6])]})
    fgen1 = flattenedFeatureGenerator(FeatureGeneratorTakeColumns("a"))
    assert fgen1.generate(inputDf).equals(pd.DataFrame({"a_0": [1], "a_1": [2]}))


class TestFgen(FeatureGenerator):
    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame = None, ctx=None):
        pass

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        return df


class RuleBasedTestFgen(RuleBasedFeatureGenerator):
    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        return df


class TestFgenBasics:
    def __init__(self):
        self.testdf = pd.DataFrame({"foo": [1, 2], "bar": [1, 2]})

    def test_basicProperties(self):
        testfgen = TestFgen()

        assert not testfgen.isFitted()
        assert testfgen.getGeneratedColumnNames() is None
        testfgen.fit(self.testdf)
        assert testfgen.isFitted()
        testfgen.generate(self.testdf)
        assert set(testfgen.getGeneratedColumnNames()) == {"foo", "bar"}
    
    @pytest.mark.parametrize("fgen", [TestFgen(), RuleBasedTestFgen(), MultiFeatureGenerator(TestFgen()),
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

