import numpy as np
import pandas as pd
import pytest

from sensai.featuregen import FeatureGeneratorFlattenColumns, FeatureGeneratorTakeColumns, getFlattenedFeatureGenerator


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
    fgen1 = getFlattenedFeatureGenerator(FeatureGeneratorTakeColumns("a"))
    assert fgen1.generate(inputDf).equals(pd.DataFrame({"a_0": [1], "a_1": [2]}))
