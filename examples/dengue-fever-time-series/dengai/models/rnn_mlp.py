from typing import Union, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..data import COLS_FEATURES
from ..features import COL_FGEN_HISTORY_WINDOW, FeatureGeneratorHistoryWeekWindow
from ..vectorisers import AutoregressiveResultHandler, \
    create_history_sequence_vectoriser
from sensai.data_transformation import SkLearnTransformerFactoryFactory, DFTSkLearnTransformer
from sensai.torch import TorchVectorRegressionModel, TorchModel, NNOptimiserParams, Tensoriser
from sensai.torch.torch_models.mlp.mlp_modules import MultiLayerPerceptron


class RnnMlp(TorchVectorRegressionModel):
    def __init__(self, nn_optimiser_params: NNOptimiserParams,
            rnn_hidden_dim: int,
            mlp_hidden_dims: Sequence[int], mlp_dropout: float,
            week_window_size: int,
            use_rnn=True,
            feature_columns: Optional[List[str]] = None,
            auto_regressive=False):
        super().__init__(self._create_torch_model, nn_optimiser_params=nn_optimiser_params)
        self.window_size = week_window_size
        self.use_rnn = use_rnn
        self.rnn_hidden_dim = rnn_hidden_dim
        self.mlp_dropout = mlp_dropout
        self.mlp_hidden_dims = mlp_hidden_dims
        self.use_positional_encoding = False
        if feature_columns is None:
            feature_columns = COLS_FEATURES
        target_transformer_factory = SkLearnTransformerFactoryFactory.MaxAbsScaler()
        autoregressive_result_handler: Optional[AutoregressiveResultHandler] = None
        if auto_regressive:
            autoregressive_result_handler = AutoregressiveResultHandler()
            self.with_autoregressive_result_handler(autoregressive_result_handler)
        self.history_vectoriser = create_history_sequence_vectoriser(feature_columns, auto_regressive,
            target_transformer_factory, autoregressive_result_handler)
        self.time_series_dim: Optional[int] = None
        self.with_feature_generator(FeatureGeneratorHistoryWeekWindow(week_window_size))
        self.with_input_tensoriser(self.InputTensoriser(self))
        self.with_target_transformer(DFTSkLearnTransformer(target_transformer_factory()))

    def _create_torch_model(self):
        return self.TorchModel(self)

    class InputTensoriser(Tensoriser):
        def __init__(self, model: "RnnMlp"):
            self.model = model

        def fit(self, df: pd.DataFrame, model=None):
            windows = df[COL_FGEN_HISTORY_WINDOW]
            self.model.history_vectoriser.fit(windows)
            dim = model.history_vectoriser.get_vector_dim(windows.iloc[0])
            if self.model.use_positional_encoding:
                dim += 1
            self.model.time_series_dim = dim

        def _tensorise(self, df: pd.DataFrame) -> Union[torch.Tensor, List[torch.Tensor]]:
            vector_seq_list, _ = self.model.history_vectoriser.apply_multi(df[COL_FGEN_HISTORY_WINDOW])
            x = torch.tensor(np.stack(vector_seq_list), dtype=torch.float)

            if self.model.use_positional_encoding:
                pos = torch.tensor(np.linspace(0, 1, self.model.window_size), dtype=torch.float) \
                    .view(self.model.window_size, 1)
                pos = pos.repeat(x.shape[0], 1).view(x.shape[0], self.model.window_size, 1)
                x = torch.cat([x, pos], dim=-1)

            return x

    class TorchModel(TorchModel):
        def __init__(self, parent: "RnnMlp"):
            super().__init__(cuda=False)
            self.parent = parent

        def create_torch_module(self) -> torch.nn.Module:
            return self.RnnMlpModule(self.parent)

        class RnnMlpModule(torch.nn.Module):
            def __init__(self, parent: "RnnMlp"):
                super().__init__()
                self.use_rnn = parent.use_rnn
                self.lstm = torch.nn.LSTM(input_size=parent.time_series_dim, hidden_size=parent.rnn_hidden_dim, batch_first=True)
                mlp_input_dim = parent.rnn_hidden_dim if self.use_rnn else parent.time_series_dim * parent.window_size
                self.mlp = MultiLayerPerceptron(mlp_input_dim, 1, hidden_dims=parent.mlp_hidden_dims,
                    hid_activation_fn=torch.relu, output_activation_fn=None, p_dropout=parent.mlp_dropout)

            def forward(self, x):
                if self.use_rnn:
                    _, (l, _) = self.lstm(x)
                    l = l.squeeze(0)
                else:
                    l = x.view(x.shape[0], x.shape[1] * x.shape[2])
                return self.mlp(l)
