import logging

import numpy as np
from typing_extensions import Protocol

log = logging.getLogger(__name__)


class SklearnTransformerProtocol(Protocol):
    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        pass

    def transform(self, arr: np.ndarray) -> np.ndarray:
        pass

    def fit(self, arr: np.ndarray):
        pass
