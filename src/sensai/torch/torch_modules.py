"""
This module exists only for backward compatibility with object that were pickled by earlier versions of sensAI.
"""

from .torch_base import MCDropoutCapableNNModule
from .torch_models.mlp.mlp_modules import MultiLayerPerceptron
from .torch_models.lstnet.lstnet_modules import LSTNetwork
from .torch_models.residualffn.residualffn_modules import ResidualFeedForwardNetwork
from ..util import mark_used

mark_used(MCDropoutCapableNNModule, MultiLayerPerceptron, LSTNetwork, ResidualFeedForwardNetwork)