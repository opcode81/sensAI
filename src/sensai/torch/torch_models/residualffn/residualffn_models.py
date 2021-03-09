import logging
from typing import Sequence, Union

import torch

from .residualffn_modules import ResidualFeedForwardNetwork
from ...torch_base import VectorTorchModel, TorchVectorRegressionModel
from ...torch_opt import NNOptimiserParams
from ....normalisation import NormalisationMode

log = logging.getLogger(__name__)


class ResidualFeedForwardNetworkTorchModel(VectorTorchModel):

    def __init__(self, cuda, hiddenDims: Sequence[int], bottleneckDimensionFactor: float = 1, pDropout=None,
            useBatchNormalisation: bool = False):
        super().__init__(cuda=cuda)
        self.hiddenDims = hiddenDims
        self.bottleneckDimensionFactor = bottleneckDimensionFactor
        self.pDropout = pDropout
        self.useBatchNormalisation = useBatchNormalisation

    def createTorchModuleForDims(self, inputDim, outputDim) -> torch.nn.Module:
        return ResidualFeedForwardNetwork(inputDim, outputDim, self.hiddenDims, self.bottleneckDimensionFactor,
            pDropout=self.pDropout, useBatchNormalisation=self.useBatchNormalisation)


class ResidualFeedForwardNetworkVectorRegressionModel(TorchVectorRegressionModel):

    def __init__(self, hiddenDims: Sequence[int], bottleneckDimensionFactor: float = 1, cuda=True, pDropout=None,
            useBatchNormalisation: bool = False, normalisationMode=NormalisationMode.NONE,
            nnOptimiserParams: Union[NNOptimiserParams, dict] = None):
        super().__init__(ResidualFeedForwardNetworkTorchModel, [cuda, hiddenDims],
            dict(bottleneckDimensionFactor=bottleneckDimensionFactor, pDropout=pDropout, useBatchNormalisation=useBatchNormalisation),
            normalisationMode=normalisationMode, nnOptimiserParams=nnOptimiserParams)
