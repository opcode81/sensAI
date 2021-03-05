import logging

import torch

from .mlp_modules import MultiLayerPerceptron
from ...torch_base import VectorTorchModel, TorchVectorRegressionModel, TorchVectorClassificationModel
from ...torch_opt import NNOptimiserParams
from .... import NormalisationMode

log = logging.getLogger(__name__)


class MultiLayerPerceptronTorchModel(VectorTorchModel):
    def __init__(self, cuda, hiddenDims, hidActivationFunction, outputActivationFunction, pDropout=None):
        super().__init__(cuda=cuda)
        self.hidActivationFunction = hidActivationFunction
        self.outputActivationFunction = outputActivationFunction
        self.hiddenDims = hiddenDims
        self.pDropout = pDropout

    def __str__(self):
        return f"_MLP[hiddenDims={self.hiddenDims}, hidAct={self.hidActivationFunction.__name__}, outAct={self.outputActivationFunction.__name__ if self.outputActivationFunction is not None else None}, pDropout={self.pDropout}]"

    def createTorchModuleForDims(self, inputDim, outputDim):
        return MultiLayerPerceptron(inputDim, outputDim, self.hiddenDims,
            hidActivationFn=self.hidActivationFunction, outputActivationFn=self.outputActivationFunction,
            pDropout=self.pDropout)


class MultiLayerPerceptronVectorRegressionModel(TorchVectorRegressionModel):
    def __init__(self, hiddenDims=(5, 5), hidActivationFunction=torch.sigmoid, outputActivationFunction=None,
            normalisationMode=NormalisationMode .MAX_BY_COLUMN,
            cuda=True, pDropout: float = None, nnOptimiserParams: NNOptimiserParams = None, **nnOptimiserDictParams):
        """
        :param hiddenDims: sequence containing the number of neurons to use in hidden layers
        :param hidActivationFunction: the activation function (torch.*) to use for all hidden layers
        :param outputActivationFunction: the output activation function (torch.* or None)
        :param normalisationMode: the normalisation mode to apply to input and output data
        :param cuda: whether to use CUDA (GPU acceleration)
        :param pDropout: the probability with which to apply dropouts after each hidden layer
        :param nnOptimiserParams: parameters for NNOptimiser; if None, use default (or what is specified in nnOptimiserDictParams)
        :param nnOptimiserDictParams: [for backward compatibility] parameters for NNOptimiser (alternative to nnOptimiserParams)
        """
        nnOptimiserParams = NNOptimiserParams.fromEitherDictOrInstance(nnOptimiserDictParams, nnOptimiserParams)
        super().__init__(MultiLayerPerceptronTorchModel, [cuda, hiddenDims, hidActivationFunction, outputActivationFunction],
                dict(pDropout=pDropout), normalisationMode, nnOptimiserParams)


class MultiLayerPerceptronVectorClassificationModel(TorchVectorClassificationModel):
    def __init__(self, hiddenDims=(5, 5), hidActivationFunction=torch.sigmoid, outputActivationFunction=torch.sigmoid,
            normalisationMode=NormalisationMode.MAX_BY_COLUMN, cuda=True, pDropout=None, nnOptimiserParams: NNOptimiserParams = None,
            **nnOptimiserDictParams):
        """
        :param hiddenDims: sequence containing the number of neurons to use in hidden layers
        :param hidActivationFunction: the activation function (torch.*) to use for all hidden layers
        :param outputActivationFunction: the output activation function (torch.*)
        :param normalisationMode: the normalisation mode to apply to input and output data
        :param cuda: whether to use CUDA (GPU acceleration)
        :param pDropout: the probability with which to apply dropouts after each hidden layer
        :param nnOptimiserParams: parameters for NNOptimiser; if None, use default (or what is specified in nnOptimiserDictParams)
        :param nnOptimiserDictParams: [for backward compatibility] parameters for NNOptimiser (alternative to nnOptimiserParams)
        """
        nnOptimiserParams = NNOptimiserParams.fromEitherDictOrInstance(nnOptimiserDictParams, nnOptimiserParams)
        super().__init__(MultiLayerPerceptronTorchModel, [cuda, hiddenDims, hidActivationFunction, outputActivationFunction],
            dict(pDropout=pDropout), normalisationMode, nnOptimiserParams)
