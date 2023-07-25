import random
from copy import copy
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from sensai import InputOutputData
from sensai.data_transformation import DFTDRowFilterOnIndex, \
    InvertibleDataFrameTransformer, DFTDropNA
from sensai.featuregen import FeatureGeneratorTakeColumns, FeatureGenerator
from sensai.vector_model import RuleBasedVectorRegressionModel, VectorRegressionModel, VectorClassificationModel


class FittableFgen(FeatureGenerator):

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame = None, ctx=None):
        pass

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        return df


class FittableDFT(InvertibleDataFrameTransformer):

    def apply_inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _fit(self, df: pd.DataFrame):
        pass

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class SampleRuleBasedVectorModel(RuleBasedVectorRegressionModel):
    def __init__(self):
        super(SampleRuleBasedVectorModel, self).__init__(predicted_variable_names=["prediction"])

    def _predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"prediction": 1}, index=X.index)

    def get_predicted_variable_names(self):
        return ["prediction"]

    def is_regression_model(self) -> bool:
        return True


class SampleVectorModel(VectorRegressionModel):

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"prediction": 1}, index=x.index)

    def _fit(self, x: pd.DataFrame, y: Optional[pd.DataFrame]):
        pass


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
    return SampleRuleBasedVectorModel()


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
    @pytest.mark.parametrize("model", [SampleRuleBasedVectorModel(), fittedVectorModel()])
    def test_isFittedWhenPreprocessorsRuleBased(self, model, ruleBasedDFT, ruleBasedFgen):
        assert model.is_fitted()
        model.with_feature_generator(ruleBasedFgen)
        model.with_input_transformers(ruleBasedDFT)
        assert model.is_fitted()

    @pytest.mark.parametrize("modelConstructor", [SampleRuleBasedVectorModel, fittedVectorModel])
    def test_isFittedWithFittableProcessors(self, modelConstructor, fittableDFT, fittableFgen):
        # is fitted after fit with model
        model = modelConstructor().with_raw_input_transformers(fittableDFT)
        assert not model.is_fitted()
        model.fit(testX, testY)
        assert model.is_fitted()
        
        # is fitted if DFT is fitted
        fittedDFT = copy(fittableDFT)
        fittedDFT.fit(testX)
        model = modelConstructor().with_raw_input_transformers(fittedDFT)
        assert model.is_fitted()

        # same for fgen
        model = modelConstructor().with_feature_generator(fittableFgen)
        assert not model.is_fitted()
        model.fit(testX, testY)
        assert model.is_fitted()

        fittedFgen = copy(fittableFgen)
        fittedFgen.fit(testX)
        model = modelConstructor().with_feature_generator(fittedFgen)
        assert model.is_fitted()

    def test_isFittedWithTargetTransformer(self, vectorModel, fittableDFT):
        assert not vectorModel.is_fitted()
        vectorModel.fit(testX, testY)
        assert vectorModel.is_fitted()
        vectorModel.with_target_transformer(fittableDFT)

        # test fitting separately
        assert not vectorModel.is_fitted()
        vectorModel.get_target_transformer().fit(testX)
        assert vectorModel.is_fitted()

        # test fitting together
        vectorModel = SampleVectorModel().with_target_transformer(FittableDFT())
        assert not vectorModel.is_fitted()
        vectorModel.fit(testX, testY)
        assert vectorModel.is_fitted()
        assert vectorModel.get_target_transformer().is_fitted()


def test_InputRowsRemovedByTransformer(irisClassificationTestCase):
    """
    Tests handling of case where the input generation process removes rows from the raw data
    """
    iodata = irisClassificationTestCase.data

    # create some random N/A values in one of the columns
    numNAValues = 20
    inputs = iodata.inputs.copy()
    rand = random.Random(42)
    fullIndices = list(range(len(inputs)))
    indices = rand.sample(fullIndices, numNAValues)
    inputs.iloc[:, 0].iloc[indices] = np.nan
    iodata = InputOutputData(inputs, iodata.outputs)
    expectedLength = len(iodata) - numNAValues

    class MyModel(VectorClassificationModel):
        def _fit_classifier(self, x: pd.DataFrame, y: pd.DataFrame):
            assert len(x) == expectedLength
            assert len(y) == expectedLength
            assert all(x.index.values == y.index.values)

        def _predict_class_probabilities(self, x: pd.DataFrame) -> pd.DataFrame:
            pass

    model = MyModel().with_raw_input_transformers(DFTDropNA())
    model.fit(iodata.inputs, iodata.outputs)
