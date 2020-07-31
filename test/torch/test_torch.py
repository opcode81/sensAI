import re

import sklearn
import torch

import sensai.torch
from sensai import NormalisationMode
from sensai.data_transformation import DFTNormalisation


def test_MLPClassifier(irisDataSet, irisClassificationTestCase):
    featureNames = irisDataSet.getInputOutputData().inputs.columns
    dftNorm = DFTNormalisation([DFTNormalisation.Rule(re.escape(f)) for f in featureNames], defaultTransformerFactory=sklearn.preprocessing.StandardScaler)
    model = sensai.torch.models.MultiLayerPerceptronVectorClassificationModel(hiddenDims=(50,25,8), cuda=False, epochs=100, optimiser="adam",
        batchSize=200, normalisationMode=NormalisationMode.NONE, hidActivationFunction=torch.tanh).withName("torchMLPClassifier").withInputTransformers([dftNorm])
    irisClassificationTestCase.testMinAccuracy(model, 0.8)
