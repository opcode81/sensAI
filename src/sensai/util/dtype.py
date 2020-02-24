from typing import Union

import logging
import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def toFloatArray(data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    if type(data) is np.ndarray:
        values = data
    elif type(data) is pd.DataFrame:
        values = data.values
    if values.dtype == "object":
        _log.warning("Input array of dtype 'object' will be converted to float64 - this is potentially unsafe!")
        values = values.astype("float64", copy=False)
    return values
