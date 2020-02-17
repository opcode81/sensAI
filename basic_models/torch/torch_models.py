import logging

import torch

from . import torch_modules
from .torch_base import WrappedTorchVectorModule, TorchVectorRegressionModel, TorchVectorClassificationModel
from ..normalisation import NormalisationMode

log = logging.getLogger(__name__)


class MultiLayerPerceptron(WrappedTorchVectorModule):
    def __init__(self, cuda, hiddenDims, hidActivationFunction, outputActivationFunction, pDropout=None):
        super().__init__(cuda=cuda)
        self.hidActivationFunction = hidActivationFunction
        self.outputActivationFunction = outputActivationFunction
        self.hiddenDims = hiddenDims
        self.pDropout = pDropout

    def __str__(self):
        return f"_MLP[hiddenDims={self.hiddenDims}, hidAct={self.hidActivationFunction.__name__}, outAct={self.outputActivationFunction.__name__ if self.outputActivationFunction is not None else None}, pDropout={self.pDropout}]"

    def createTorchVectorModule(self, inputDim, outputDim):
        return torch_modules.MultiLayerPerceptronModule(inputDim, outputDim, self.hiddenDims,
            hidActivationFn=self.hidActivationFunction, outputActivationFn=self.outputActivationFunction,
            pDropout=self.pDropout)


class TorchMultiLayerPerceptronVectorRegressionModel(TorchVectorRegressionModel):
    """
    HVS model which uses a torch implementation of a multi-layer perceptron
    """

    log = log.getChild(__qualname__)

    def __init__(self, hiddenDims=(5, 5), hidActivationFunction=torch.sigmoid, outputActivationFunction=None,
            normalisationMode=NormalisationMode.MAX_BY_COLUMN,
            cuda=True, pDropout=None, **nnOptimiserParams):
        """
        :param hiddenDims: sequence containing the number of neurons to use in hidden layers
        :param hidActivationFunction: the activation function (torch.*) to use for all hidden layers
        :param outputActivationFunction: the output activation function (torch.* or None)
        :param normalisationMode: the normalisation mode to apply to input and output data
        :param cuda: whether to use CUDA (GPU acceleration)
        :param pDropout: the probability with which to apply dropouts after each hidden layer
        :param nnOptimiserParams: parameters to pass on to NNOptimiser
        """
        super().__init__(MultiLayerPerceptron, [cuda, hiddenDims, hidActivationFunction, outputActivationFunction],
                dict(pDropout=pDropout), normalisationMode, nnOptimiserParams)


class TorchMultiLayerPerceptronVectorClassificationModel(TorchVectorClassificationModel):
    """
    HVS model which uses a torch implementation of a multi-layer perceptron
    """

    log = log.getChild(__qualname__)

    def __init__(self, hiddenDims=(5, 5), hidActivationFunction=torch.sigmoid, outputActivationFunction=torch.sigmoid,
            normalisationMode=NormalisationMode.MAX_BY_COLUMN, cuda=True, pDropout=None,
            **nnOptimiserParams):
        """
        :param hiddenDims: sequence containing the number of neurons to use in hidden layers
        :param hidActivationFunction: the activation function (torch.*) to use for all hidden layers
        :param outputActivationFunction: the output activation function (torch.*)
        :param normalisationMode: the normalisation mode to apply to input and output data
        :param cuda: whether to use CUDA (GPU acceleration)
        :param pDropout: the probability with which to apply dropouts after each hidden layer
        :param nnOptimiserParams: parameters to pass on to NNOptimiser
        """
        super().__init__(MultiLayerPerceptron, [cuda, hiddenDims, hidActivationFunction, outputActivationFunction],
            dict(pDropout=pDropout), normalisationMode, nnOptimiserParams)
