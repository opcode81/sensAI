import logging
from typing import Optional, List, Union

import pandas as pd
import torch

from ....torch import TorchVectorRegressionModel, Tensoriser, NNOptimiserParams, TorchModel
from ....torch.torch_models.seq.seq_modules import EncoderDecoderModule, EncoderFactory, DecoderFactory
from ....vectoriser import SequenceVectoriser

log = logging.getLogger(__name__)


class EncoderDecoderVectorRegressionModel(TorchVectorRegressionModel):
    def __init__(self, cuda: bool,
            history_sequence_column_name: str,
            history_sequence_vectoriser: SequenceVectoriser,
            history_sequence_variable_length: bool,
            target_sequence_column_name: str,
            target_sequence_vectoriser: SequenceVectoriser,
            latent_dim: int,
            encoder_factory: EncoderFactory,
            decoder_factory: DecoderFactory,
            nn_optimiser_params: Optional[NNOptimiserParams] = None):
        """
        :param cuda: whether to use a CUDA device
        :param history_sequence_column_name:
        :param history_sequence_vectoriser:
        :param history_sequence_variable_length:
        :param target_sequence_column_name: the column containing the target item sequence; Note that the column must
            contain sequences even if there is but a single target item for which predictions shall be made.
            In such cases, simply use a column that contains lists with a single item each.
        :param target_sequence_vectoriser: the vectoriser for the generation of feature vectors for the target
            items.
        :param latent_dim:
        :param encoder_factory:
        :param decoder_factory:
        :param nn_optimiser_params:
        """
        super().__init__(self._create_model, nn_optimiser_params=nn_optimiser_params)
        self.history_sequence_variable_length = history_sequence_variable_length
        self.latent_dim = latent_dim
        self.cuda = cuda
        self.decoder_factory = decoder_factory
        self.encoder_factory = encoder_factory
        self.sequenceVectoriser = history_sequence_vectoriser
        self.history_sequence_dim_per_item: Optional[int] = None
        self.target_feature_dim: Optional[int] = None
        self.with_input_tensoriser(self.InputTensoriser(history_sequence_column_name, history_sequence_vectoriser,
            target_sequence_column_name, target_sequence_vectoriser))

    def _create_model(self):
        return self.EncoderDecoderModel(self)

    class InputTensoriser(Tensoriser):
        def __init__(self,
                history_sequence_column_name: str,
                history_sequence_vectoriser: SequenceVectoriser,
                target_sequence_column_name: str,
                target_sequence_vectoriser: SequenceVectoriser):
            self.history_sequence_column_name = history_sequence_column_name
            self.history_sequence_vectoriser = history_sequence_vectoriser
            self.target_sequence_column_name = target_sequence_column_name
            self.target_sequence_vectoriser = target_sequence_vectoriser

        def fit(self, df: pd.DataFrame, model=None):
            model: "EncoderDecoderVectorRegressionModel"
            self.history_sequence_vectoriser.fit(df[self.history_sequence_column_name])
            model.history_sequence_dim_per_item = self.history_sequence_vectoriser.get_vector_dim(df[self.history_sequence_column_name].iloc[0])
            self.target_sequence_vectoriser.fit(df[self.target_sequence_column_name])
            model.target_feature_dim = self.target_sequence_vectoriser.get_vector_dim(df[self.target_sequence_column_name].iloc[0])

        def _tensorise(self, df: pd.DataFrame) -> Union[torch.Tensor, List[torch.Tensor]]:
            log.debug(f"Applying {self} to data frame of length {len(df)} ...")
            history = df[self.history_sequence_column_name]
            history_sequences, history_sequence_lengths = self.history_sequence_vectoriser.apply_multi_with_padding(history)
            targets = df[self.target_sequence_column_name]
            target_sequences, target_sequence_lengths = self.target_sequence_vectoriser.apply_multi_with_padding(targets)
            return [
                torch.tensor(history_sequences).float(),
                torch.tensor(history_sequence_lengths),
                torch.tensor(target_sequences).float(),
                torch.tensor(target_sequence_lengths),
            ]

    class EncoderDecoderModel(TorchModel):
        def __init__(self, parent: "EncoderDecoderVectorRegressionModel"):
            super().__init__(parent.cuda)
            self.parent = parent

        def create_torch_module(self) -> torch.nn.Module:
            latent_dim = self.parent.latent_dim
            target_feature_dim = self.parent.target_feature_dim
            encoder = self.parent.encoder_factory.create_encoder(self.parent.history_sequence_dim_per_item, latent_dim)
            decoder = self.parent.decoder_factory.create_decoder(latent_dim, target_feature_dim)
            return EncoderDecoderModule(encoder, decoder, self.parent.history_sequence_variable_length)
