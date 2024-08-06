from typing import Union

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from ..data import COL_TARGET, COL_FEATURE_REANALYSIS_SPEC_HUMIDITY, COL_FEATURE_REANALYSIS_DEW_POINT_K, \
    COL_FEATURE_STATION_MIN_TEMP, COL_FEATURE_STATION_AVG_TEMP
from ..features import FeatureGeneratorTakeImputedColumns
from sensai import VectorRegressionModel


class GeneralisedLinearModel(VectorRegressionModel):
    """
    The generalised linear model (GLM) from the DrivenData benchmark notebook.
    See: https://drivendata.co/blog/dengue-benchmark/
    """
    def __init__(self, alpha=1e-8):
        super().__init__()
        self.alpha = alpha
        self.model_formula = f"{COL_TARGET} ~ 1 + " \
            f"{COL_FEATURE_REANALYSIS_SPEC_HUMIDITY} + " \
            f"{COL_FEATURE_REANALYSIS_DEW_POINT_K} + " \
            f"{COL_FEATURE_STATION_MIN_TEMP} + " \
            f"{COL_FEATURE_STATION_AVG_TEMP}"
        self.with_feature_generator(FeatureGeneratorTakeImputedColumns()) \
            .with_name("GLM-DD-Benchmark")

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame):
        data = pd.concat([x, y], axis=1)
        model = smf.glm(formula=self.model_formula,
            data=data,
            family=sm.families.NegativeBinomial(alpha=self.alpha))
        self.model = model.fit()

    def _predict(self, x: pd.DataFrame) -> Union[pd.DataFrame, list]:
        result = self.model.predict(x)
        return result.to_list()
