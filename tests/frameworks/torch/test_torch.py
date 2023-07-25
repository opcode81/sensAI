import re
import os

import sklearn
import torch

import sensai.torch
from sensai import NormalisationMode, normalisation
from sensai.data_transformation import DFTNormalisation
from sensai.torch import NNOptimiser
from sensai.torch.torch_base import TorchModelFromModuleFactory
from sensai.torch.torch_data import TorchDataSetFromTensors
from sensai.torch.torch_modules import MultiLayerPerceptron
from sensai.torch.torch_opt import NNLossEvaluatorClassification, NNOptimiserParams
from sensai.featuregen import FeatureGeneratorTakeColumns


def test_MLPClassifier(irisDataSet, irisClassificationTestCase, testResources):
    featureNames = irisDataSet.getInputOutputData().inputs.columns
    dftNorm = DFTNormalisation([DFTNormalisation.Rule(re.escape(f)) for f in featureNames],
        default_transformer_factory=sklearn.preprocessing.StandardScaler)
    nn_optimiser_params = NNOptimiserParams(epochs=100, optimiser="adam", batch_size=200)
    model = sensai.torch.models.MultiLayerPerceptronVectorClassificationModel(
            hidden_dims=(50,25,8), cuda=False,
            normalisation_mode=NormalisationMode.NONE, hid_activation_function=torch.tanh,
            nn_optimiser_params=nn_optimiser_params) \
        .with_name("torchMLPClassifier") \
        .with_input_transformers([dftNorm]) \
        .with_feature_generator(FeatureGeneratorTakeColumns())
    irisClassificationTestCase.testMinAccuracy(model, 0.8)


def test_NNOptimiserWithoutValidation_MLPClassifier(irisDataSet):
    """
    Tests and demonstrates the use of NNOptimiser without validation on the iris data set, using low-level
    interfaces (i.e. without high-level abstractions such as VectorModel)
    """
    iodata = irisDataSet.getInputOutputData()
    outputs = iodata.outputs.iloc[:, 0]
    classLabels = list(outputs.unique())
    outputTensor = torch.tensor([classLabels.index(l) for l in outputs])
    scaler = normalisation.VectorDataScaler(iodata.inputs, normalisation.NormalisationMode.MAX_BY_COLUMN)
    inputTensor = torch.tensor(scaler.get_normalised_array(iodata.inputs), dtype=torch.float32)
    dataSet = TorchDataSetFromTensors(inputTensor, outputTensor, False)
    model = TorchModelFromModuleFactory(lambda: MultiLayerPerceptron(inputTensor.shape[1], len(classLabels),
            (4, 3), hid_activation_fn=torch.tanh, output_activation_fn=None), cuda=False)
    NNOptimiser(NNOptimiserParams(loss_evaluator=NNLossEvaluatorClassification(NNLossEvaluatorClassification.LossFunction.CROSSENTROPY), train_fraction=1.0, epochs=300,
            optimiser="adam")).fit(model, dataSet)
    modelOutputs = model.apply(inputTensor, as_numpy=False)
    accuracy = torch.sum(torch.argmax(modelOutputs, 1) == outputTensor).item() / len(outputTensor)
    assert accuracy > 0.9