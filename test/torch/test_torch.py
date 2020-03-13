import logging
import re

import sklearn
import torch

from sensai import NormalisationMode
from sensai.data_transformation import DFTNormalisation
from ..classification import IrisClassificationTestCase, IrisDataSet
import sensai.torch


def test_MLPClassifier():
    featureNames = IrisDataSet.getInputOutputData().inputs.columns
    dftNorm = DFTNormalisation([DFTNormalisation.Rule(re.escape(f)) for f in featureNames], defaultTransformerFactory=sklearn.preprocessing.StandardScaler)
    model = sensai.torch.models.TorchMultiLayerPerceptronVectorClassificationModel(hiddenDims=(50,25,8), cuda=False, epochs=1000, optimiser="adam",
        batchSize=200, normalisationMode=NormalisationMode.NONE, hidActivationFunction=torch.tanh).withName("torchMLPClassifier").withInputTransformers([dftNorm])
    IrisClassificationTestCase().testMinAccuracy(model, 0.9)
