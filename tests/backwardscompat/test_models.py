import os
from glob import glob

import pytest
import sklearn

import sensai
from sensai import VectorModel
from sensai.util.pickle import load_pickle
from model_test_case import RESOURCE_DIR
from sensai.util.version import Version


def test_classification_model_backward_compatibility_v0_0_4(testResources, irisClassificationTestCase):
    # The model file was generated with tests/frameworks/torch/test_torch.test_MLPClassifier at commit f93c6b11d
    modelPath = os.path.join(testResources, "torch_mlp.pickle")
    model = VectorModel.load(modelPath)
    assert isinstance(model, sensai.torch.models.MultiLayerPerceptronVectorClassificationModel)
    irisClassificationTestCase.testMinAccuracy(model, 0.8, fit=False)


@pytest.mark.parametrize("pickle_file", glob(f"{RESOURCE_DIR}/backward_compatibility/regression_model_*.v0.2.0.pickle"))
def test_regression_model_backward_compatibility_v0_2_0(pickle_file, diabetesRegressionTestCase):
    """
    Tests for compatibility with models created with v0.2.0 using create_test_models.py
    """
    sklearn_version = Version(sklearn)
    if not sklearn_version.is_at_most(1, 0, 2):
        # NOTE: sklearn is breaking persisted objects with newer versions (e.g. OneHotEncoder),
        # so we can only run this test with the correct sklearn version
        pytest.skip("This test is only compatible with sklearn <= 1.0.2")
    d = load_pickle(pickle_file)
    r2, model = d["R2"], d["model"]
    diabetesRegressionTestCase.testMinR2(model, r2-0.02, fit=False)

