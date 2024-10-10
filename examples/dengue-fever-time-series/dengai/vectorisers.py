from typing import Optional, List, Callable

import numpy as np
import pandas as pd

from .data import WeekDataPointInWindow, COL_TARGET, COLS_FEATURES_STD, COLS_FEATURES_NONSTD
from sensai.data_transformation import SkLearnTransformerFactoryFactory
from sensai.torch import TorchAutoregressiveResultHandler
from sensai.vectoriser import Vectoriser, SequenceVectoriser


class InputFeatureVectoriser(Vectoriser[WeekDataPointInWindow]):
    def __init__(self, col: str, transformer):
        self.col = col
        super().__init__(self._get_value, transformer=transformer)

    def _get_value(self, w: WeekDataPointInWindow) -> float:
        return w.week_data_point.get_value(self.col, imputed=True)


class AutoregressiveResultHandler(TorchAutoregressiveResultHandler):
    def __init__(self):
        self.results = {}
        self.min_index = None

    def clear_results(self):
        self.__init__()

    def save_results(self, input_df: pd.DataFrame, results: np.ndarray) -> None:
        for t, result_arr in zip(input_df.itertuples(), results):
            if self.min_index is None:
                self.min_index = t.Index
            self.results[t.Index] = result_arr[0]

    def get_result(self, w: WeekDataPointInWindow) -> Optional[float]:
        index = w.week_data_point.get_index()
        result = self.results.get(index)
        if self.min_index is not None and index >= self.min_index and result is None:
            raise AssertionError("Expected result was not previously stored")
        return result


class AutoregressiveFeatureVectoriser(Vectoriser[WeekDataPointInWindow]):
    def __init__(self, transformer, current_value=-3,
            result_handler: Optional[AutoregressiveResultHandler] = None):
        super().__init__(self._get_value, transformer=transformer)
        self.current_value = current_value
        self.result_handler = result_handler

    def _get_value(self, w: WeekDataPointInWindow) -> float:
        if w.is_current():
            return self.current_value
        else:
            if self.result_handler is not None:
                value = self.result_handler.get_result(w)
                if value is not None:
                    return value
            return w.week_data_point.get_value(COL_TARGET)


def create_history_sequence_vectoriser(feature_columns: List[str], auto_regressive: bool, target_transformer_factory: Callable,
        autoregressive_result_handler: Optional[AutoregressiveResultHandler] = None):
    vectorisers = []
    if auto_regressive:
        vectorisers.append(AutoregressiveFeatureVectoriser(target_transformer_factory(),
            result_handler=autoregressive_result_handler))
    std_transformer_factory = SkLearnTransformerFactoryFactory.StandardScaler()
    vectorisers.extend([InputFeatureVectoriser(col, std_transformer_factory()) for col in feature_columns
        if col in COLS_FEATURES_STD])
    nonstd_transformer_factory = SkLearnTransformerFactoryFactory.MaxAbsScaler()
    vectorisers.extend([InputFeatureVectoriser(col, nonstd_transformer_factory()) for col in feature_columns
        if col in COLS_FEATURES_NONSTD])
    return SequenceVectoriser(vectorisers, fitting_mode=SequenceVectoriser.FittingMode.UNIQUE,
        unique_id_provider=WeekDataPointInWindow.IdentifierProvider())


def create_target_sequence_vectoriser(history_sequence_vectoriser: SequenceVectoriser):
    return SequenceVectoriser([v for v in history_sequence_vectoriser.vectorisers if not isinstance(v, AutoregressiveFeatureVectoriser)],
        unique_id_provider=WeekDataPointInWindow.IdentifierProvider(),
        refit_vectorisers=False)
