from typing import Union, Optional

import pandas as pd

from ..data import COL_WEEK_OF_YEAR, COL_TARGET
from sensai import VectorRegressionModel


class MeanPastYearWeekModel(VectorRegressionModel):
    def __init__(self):
        super().__init__()
        self._mean_week_values: Optional[pd.Series] = None

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame):
        df = pd.concat([x, y], axis=1)
        self._mean_week_values = df[[COL_WEEK_OF_YEAR, COL_TARGET]].groupby(COL_WEEK_OF_YEAR).mean()

    def _predict(self, x: pd.DataFrame) -> Union[pd.DataFrame, list]:
        return x[COL_WEEK_OF_YEAR].apply(lambda w: self._mean_week_values.loc[w])
