import os

import sensai
from sensai import VectorModel


def test_modelCanBeLoaded(testResources, irisClassificationTestCase):
    # The model file was generated with tests/frameworks/torch/test_torch.test_MLPClassifier at commit f93c6b11d
    # NOTE: This test fails with scikit-learn 0.23 because of a change in StandardScaler
    modelPath = os.path.join(testResources, "torch_mlp.pickle")
    model = VectorModel.load(modelPath)
    assert isinstance(model, sensai.torch.models.MultiLayerPerceptronVectorClassificationModel)
    irisClassificationTestCase.testMinAccuracy(model, 0.8, fit=False)

