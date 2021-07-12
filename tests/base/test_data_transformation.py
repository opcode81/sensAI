import numpy as np
import pandas as pd
import sklearn.preprocessing

from sensai.data_transformation import DataFrameTransformer, RuleBasedDataFrameTransformer, DataFrameTransformerChain, DFTNormalisation


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
        assert testdft.info()["changeInColumnNames"] is None
        testdft.fit(self.testdf)
        assert testdft.isFitted()
        assert all(testdft.apply(self.testdf) == pd.DataFrame({"foo": [1, 2], "baz": [1, 2]}))
        assert testdft.info()["changeInColumnNames"] is not None

        # testing apply with no change in columns
        testdft.apply(pd.DataFrame({"foo": [1, 2], "baz": [1, 2]}))
        assert testdft.info()["changeInColumnNames"] == "none"

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


class TestDFTNormalisation:
    def test_multiColumnSingleRuleIndependent(self):
        arr = np.array([1, 5, 10])
        df = pd.DataFrame({"foo": arr, "bar": arr*100})
        dft = DFTNormalisation([DFTNormalisation.Rule(r"foo|bar", transformer=sklearn.preprocessing.MaxAbsScaler(), independentColumns=True)])
        df2 = dft.fitApply(df)
        assert np.all(df2.foo == arr/10) and np.all(df2.bar == arr/10)

    def test_multiColumnSingleRule(self):
        arr = np.array([1, 5, 10])
        df = pd.DataFrame({"foo": arr, "bar": arr*100})
        dft = DFTNormalisation([DFTNormalisation.Rule(r"foo|bar", transformer=sklearn.preprocessing.MaxAbsScaler(), independentColumns=False)])
        df2 = dft.fitApply(df)
        assert np.all(df2.foo == arr/1000) and np.all(df2.bar == arr/10)

    def test_arrayValued(self):
        arr = np.array([1, 5, 10])
        df = pd.DataFrame({"foo": [arr, 2*arr, 10*arr]})
        dft = DFTNormalisation([DFTNormalisation.Rule(r"foo|bar", transformer=sklearn.preprocessing.MaxAbsScaler(), arrayValued=True)])
        df2 = dft.fitApply(df)
        assert np.all(df2.foo.iloc[0] == arr/100) and np.all(df2.foo.iloc[-1] == arr/10)
