import pandas as pd

from sensai.data_transformation import DataFrameTransformer, RuleBasedDataFrameTransformer, DataFrameTransformerChain


class TestDFTTransformerBasics:
    class TestDFT(DataFrameTransformer):
        def _fit(self, df: pd.DataFrame):
            pass

        def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"foo": [1, 2], "baz": [1, 2]})

    class RuleBasedTestDFT(RuleBasedDataFrameTransformer):
        def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

    testdf = pd.DataFrame({"foo": [1, 2], "bar": [1, 2]})

    def test_basicProperties(self):
        testdft = self.TestDFT()

        assert not testdft.isFitted()
        assert testdft.getChangeInColumnNames() is None
        testdft.fit(self.testdf)
        assert testdft.isFitted()
        assert all(testdft.apply(self.testdf) == pd.DataFrame({"foo": [1, 2], "baz": [1, 2]}))
        assert testdft.getChangeInColumnNames() is not None
        assert testdft.getChangeInColumnNames()["removedColumns"] == {"bar"}
        assert testdft.getChangeInColumnNames()["addedColumns"] == {"baz"}

        # testing with no change in columns
        testdft.apply(pd.DataFrame({"foo": [1, 2], "baz": [1, 2]}))
        assert testdft.getChangeInColumnNames()["removedColumns"] == set()
        assert testdft.getChangeInColumnNames()["addedColumns"] == set()

    def test_ruleBasedAlwaysFitted(self):
        assert self.RuleBasedTestDFT().isFitted()

    def test_emptyChainFitted(self):
        assert DataFrameTransformerChain().isFitted()

    def test_combinationFittedIffEachMemberFitted(self):
        # if one of the fgens is not fitted, the combination is not fitted either
        dftChain = DataFrameTransformerChain(self.TestDFT(), self.RuleBasedTestDFT())
        assert dftChain.dataFrameTransformers[1].isFitted()
        assert not dftChain.isFitted()
        dftChain.fit(self.testdf)
        assert dftChain.isFitted()
        assert dftChain.dataFrameTransformers[0].isFitted()

        # if all fgens are fitted, the combination is also fitted, even if fit was not called
        dftChain = DataFrameTransformerChain([self.RuleBasedTestDFT(), self.RuleBasedTestDFT()])
        assert dftChain.isFitted()
