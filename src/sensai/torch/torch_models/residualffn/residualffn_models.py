import logging
from typing import Sequence, Union, Optional

import torch

from .residualffn_modules import ResidualFeedForwardNetwork
from ...torch_base import VectorTorchModel, TorchVectorRegressionModel
from ...torch_opt import NNOptimiserParams
from ....normalisation import NormalisationMode

log: logging.Logger = logging.getLogger(__name__)


class ResidualFeedForwardNetworkTorchModel(VectorTorchModel):

    def __init__(self, cuda: bool, hidden_dims: Sequence[int], bottleneck_dimension_factor: float = 1, p_dropout=None,
            use_batch_normalisation: bool = False) -> None:
        super().__init__(cuda=cuda)
        self.hiddenDims = hidden_dims
        self.bottleneckDimensionFactor = bottleneck_dimension_factor
        self.pDropout = p_dropout
        self.useBatchNormalisation = use_batch_normalisation

    def create_torch_module_for_dims(self, input_dim: int, output_dim: int) -> torch.nn.Module:
        return ResidualFeedForwardNetwork(input_dim, output_dim, self.hiddenDims, self.bottleneckDimensionFactor,
            p_dropout=self.pDropout, use_batch_normalisation=self.useBatchNormalisation)


class ResidualFeedForwardNetworkVectorRegressionModel(TorchVectorRegressionModel):

    def __init__(self,
            hidden_dims: Sequence[int],
            bottleneck_dimension_factor: float = 1,
            cuda: bool = True,
            p_dropout: Optional[float] = None,
            use_batch_normalisation: bool = False,
            normalisation_mode: NormalisationMode = NormalisationMode.NONE,
            nn_optimiser_params: Union[NNOptimiserParams, dict, None] = None) -> None:
        super().__init__(ResidualFeedForwardNetworkTorchModel, [cuda, hidden_dims],
            dict(bottleneckDimensionFactor=bottleneck_dimension_factor, pDropout=p_dropout, useBatchNormalisation=use_batch_normalisation),
            normalisation_mode=normalisation_mode, nn_optimiser_params=nn_optimiser_params)
