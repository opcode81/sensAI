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
from sensai.torch.torch_opt import NNLossEvaluatorClassification
from sensai.featuregen import FeatureGeneratorTakeColumns


def test_MLPClassifier(irisDataSet, irisClassificationTestCase, testResources):
    featureNames = irisDataSet.getInputOutputData().inputs.columns
    dftNorm = DFTNormalisation([DFTNormalisation.Rule(re.escape(f)) for f in featureNames], defaultTransformerFactory=sklearn.preprocessing.StandardScaler)
    model = sensai.torch.models.MultiLayerPerceptronVectorClassificationModel(hiddenDims=(50,25,8), cuda=False, epochs=100, optimiser="adam",
        batchSize=200, normalisationMode=NormalisationMode.NONE, hidActivationFunction=torch.tanh).withName("torchMLPClassifier").withInputTransformers([dftNorm]).withFeatureGenerator(FeatureGeneratorTakeColumns())
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
    inputTensor = torch.tensor(scaler.getNormalisedArray(iodata.inputs), dtype=torch.float32)
    dataSet = TorchDataSetFromTensors(inputTensor, outputTensor, False)
    model = TorchModelFromModuleFactory(lambda: MultiLayerPerceptron(inputTensor.shape[1], len(classLabels),
            (4, 3), hidActivationFn=torch.tanh, outputActivationFn=torch.nn.Softmax()), cuda=False)
    NNOptimiser(lossEvaluator=NNLossEvaluatorClassification(), cuda=False, trainFraction=1.0, epochs=300,
            optimiser="adam").fit(model, dataSet)
    modelOutputs = model.apply(inputTensor, asNumpy=False)
    accuracy = torch.sum(torch.argmax(modelOutputs, 1) == outputTensor).item() / len(outputTensor)
    assert accuracy > 0.9