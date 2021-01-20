import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sensai import VectorModel

log = logging.getLogger(__name__)
# we set the default level to debug because it is often interesting for the user to receive
# information debug information about shapes as data frames get converted to arrays
log.setLevel(logging.DEBUG)


class TensorModel(VectorModel, ABC):
    """
    Base class for models that input and output tensors, for examples CNNs. The fitting and predictions will still
    be performed on data frames, like in VectorModel, but now it will be expected that all entries of the
    input predictedProbaDf passed to the model are tensors of the same shape or lists of scalars. The same is expected of the ground
    truth data frame. Everything will work as well if the entries are scalars but in this case it is recommended to use
    VectorModel instead.

    If we denote the shapes of entries in the dfs as inputTensorShape and outputTensorShape,
    the model will be fit on input tensors of shape (N_rows, N_inputColumns, inputTensorShape) and output tensors of
    shape (N_rows, N_outputColumns, outputTensorShape), where empty dimensions will be stripped.
    """

    def __init__(self, checkInputColumns=True):
        super().__init__(checkInputColumns=checkInputColumns)
        self._inputDatapointShape: Optional[Tuple] = None
        self._outputDatapointShape: Optional[Tuple] = None

    @abstractmethod
    def _fitToArray(self, X: np.ndarray, Y: np.ndarray):
        pass

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        log.debug(f"Stacking input tensors from columns {X.columns} and from all rows to a single array. "
                  f"Note that all tensors need to have the same shape")
        X = _extractArray(X)
        Y = _extractArray(Y)
        self._inputDatapointShape = X[0].shape
        self._outputDatapointShape = Y[0].shape
        log.debug(f"Fitting on {len(X)} datapoints of shape {self._inputDatapointShape}. "
                  f"The ground truth are tensors of shape {self._outputDatapointShape}")
        self._fitToArray(X, Y)


def _extractArray(df: pd.DataFrame):
    """
    Extracts array from data frame. It is expected that each row corresponds to a data point and
    each column corresponds to a "channel". Moreover, all entries are expected to be arrays of the same shape
    (or scalars or sequences of the same length).
    We will refer to that shape as tensorShape.

    The output will be of shape (N_rows, N_columns, tensorShape). Thus, N_rows can be interpreted as dataset length
    (or batch size, if a single batch is passed) and N_columns can be interpreted as number of channels.
    Empty dimensions will be stripped, thus if the data frame has only one column, the array will have shape
    (N_rows, tensorShape).
    E.g. an image with three channels could equally be passed as predictedProbaDf of the type


    | ----|-----R-----|-----G-----|-----B------
    | =========================================
    | 0---|--channel--|--channel--|--channel
    | 1    ...

     or as predictedProbaDf of the type

    | ----|----image----|
    | ====================
    | 0---|--RGBArray--|
    | 1    ...

    In both cases the returned array will have shape (N_images, 3, width, height)

    :param df: data frame where each entry is an array of shape tensorShape
    :return: array of shape N_rows, N_columns, tensorShape with stripped empty dimensions
    """
    log.debug(f"Stacking tensors of shape {np.array(df.iloc[0, 0]).shape}")
    try:
        result = np.stack(df.apply(np.stack, axis=1))
    except ValueError:
        raise ValueError(f"No array can be extracted from frame of length {len(df)} with columns {list(df.columns)}. "
                         f"Make sure that all entries have the same shape")
    return result
