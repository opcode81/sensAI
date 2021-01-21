from copy import copy
from typing import Optional

import pandas as pd
import pytest

from sensai.data_transformation import DFTDRowFilterOnIndex, \
    InvertibleDataFrameTransformer
from sensai.featuregen import FeatureGeneratorTakeColumns, FeatureGenerator
from sensai.vector_model import VectorModel, RuleBasedRegressionModel


class FittableFgen(FeatureGenerator):

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame = None, ctx=None):
        pass

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        return df


class FittableDFT(InvertibleDataFrameTransformer):

    def applyInverse(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _fit(self, df: pd.DataFrame):
        pass

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class SampleRuleBasedModel(RuleBasedRegressionModel):

    def _predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"prediction": 1}, index=X.index)

    def getPredictedVariableNames(self):
        return ["prediction"]

    def isRegressionModel(self) -> bool:
        return True


class SampleVectorModel(VectorModel):

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"prediction": 1}, index=x.index)

    def _fit(self, X: pd.DataFrame, Y: Optional[pd.DataFrame]):
        pass

    def isRegressionModel(self) -> bool:
        return True


@pytest.fixture()
def ruleBasedFgen():
    return FeatureGeneratorTakeColumns()


@pytest.fixture()
def fittableFgen():
    return FittableFgen()


@pytest.fixture()
def ruleBasedDFT():
    return DFTDRowFilterOnIndex()


@pytest.fixture()
def fittableDFT():
    return FittableDFT()


@pytest.fixture()
def ruleBasedModel():
    return SampleRuleBasedModel()


@pytest.fixture()
def vectorModel():
    return SampleVectorModel()


testX = pd.DataFrame({"bar": [1, 2, 3]})
testY = pd.DataFrame({"prediction": [0, 0, 0]})


def fittedVectorModel():
    model = SampleVectorModel()
    model.fit(testX, testY)
    return model


class TestIsFitted:
    @pytest.mark.parametrize("model", [SampleRuleBasedModel(), fittedVectorModel()])
    def test_isFittedWhenPreprocessorsRuleBased(self, model, ruleBasedDFT, ruleBasedFgen):
        assert model.isFitted()
        model.withFeatureGenerator(ruleBasedFgen)
        model.withInputTransformers(ruleBasedDFT)
        assert model.isFitted()

    @pytest.mark.parametrize("modelConstructor", [SampleRuleBasedModel, fittedVectorModel])
    def test_isFittedWithFittableProcessors(self, modelConstructor, fittableDFT, fittableFgen):
        # is fitted after fit with model
        model = modelConstructor().withInputTransformers(fittableDFT)
        assert not model.isFitted()
        model.fit(testX, testY)
        assert model.isFitted()
        
        # is fitted if DFT is fitted
        fittedDFT = copy(fittableDFT)
        fittedDFT.fit(testX)
        model = modelConstructor().withInputTransformers(fittedDFT)
        assert model.isFitted()

        # same for fgen
        model = modelConstructor().withFeatureGenerator(fittableFgen)
        assert not model.isFitted()
        model.fit(testX, testY)
        assert model.isFitted()

        fittedFgen = copy(fittableFgen)
        fittedFgen.fit(testX)
        model = modelConstructor().withFeatureGenerator(fittedFgen)
        assert model.isFitted()

    def test_isFittedWithTargetTransformer(self, vectorModel, fittableDFT):
        assert not vectorModel.isFitted()
        vectorModel.fit(testX, testY)
        assert vectorModel.isFitted()
        vectorModel.withTargetTransformer(fittableDFT)

        # test fitting separately
        assert not vectorModel.isFitted()
        vectorModel.getTargetTransformer().fit(testX)
        assert vectorModel.isFitted()

        # test fitting together
        vectorModel = SampleVectorModel().withTargetTransformer(FittableDFT())
        assert not vectorModel.isFitted()
        vectorModel.fit(testX, testY)
        assert vectorModel.isFitted()
        assert vectorModel.getTargetTransformer().isFitted()



