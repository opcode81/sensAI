import logging
from typing import Optional

import pandas as pd
import xgboost

from . import InputOutputData
from .data import DataSplitter
from .sklearn.sklearn_base import AbstractSkLearnMultipleOneDimVectorRegressionModel, AbstractSkLearnVectorClassificationModel, \
    FeatureImportanceProviderSkLearnRegressionMultipleOneDim, FeatureImportanceProviderSkLearnClassification, ActualFitParams
from .util.pickle import setstate

log = logging.getLogger(__name__)


def is_xgboost_version_at_least(major: int, minor: Optional[int] = None, patch: Optional[int] = None):
    components = xgboost.__version__.split(".")
    for i, version in enumerate((major, minor, patch)):
        if version is not None:
            installed_version = int(components[i])
            if installed_version > version:
                return True
            if installed_version < version:
                return False
    return True


class XGBGradientBoostedVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel,
        FeatureImportanceProviderSkLearnRegressionMultipleOneDim):
    """
    XGBoost's regression model using gradient boosted trees
    """

    def __init__(self, random_state=42,
            early_stopping_rounds: Optional[int] = None,
            early_stopping_data_splitter: Optional[DataSplitter] = None,
            **model_args):
        """
        :param model_args: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor
        """
        super().__init__(xgboost.XGBRegressor, random_state=random_state, early_stopping_rounds=early_stopping_rounds,
            **model_args)
        self.is_early_stopping_enabled = early_stopping_rounds is not None
        self.early_stopping_data_splitter = early_stopping_data_splitter

    def __setstate__(self, state):
        setstate(XGBGradientBoostedVectorRegressionModel, self, state,
            new_default_properties=dict(
                is_early_stopping_enabled=False,
                early_stopping_data_splitter=None))

    def is_sample_weight_supported(self) -> bool:
        return True

    def _compute_actual_fit_params(self, inputs: pd.DataFrame, outputs: pd.DataFrame, weights: Optional[pd.Series] = None) -> ActualFitParams:
        kwargs = {}
        if self.is_early_stopping_enabled:
            data = InputOutputData(inputs, outputs, weights=weights)
            train_data, val_data = self.early_stopping_data_splitter.split(data)
            train_data: InputOutputData
            kwargs["eval_set"] = [(val_data.inputs, val_data.outputs)]
            inputs = train_data.inputs
            outputs = train_data.outputs
            weights = train_data.weights
            log.info(f"Early stopping enabled with validation set of size {len(val_data)}")
        params = super()._compute_actual_fit_params(inputs, outputs, weights=weights)
        params.kwargs.update(kwargs)
        return params


class XGBRandomForestVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel,
        FeatureImportanceProviderSkLearnRegressionMultipleOneDim):
    """
    XGBoost's random forest regression model
    """
    def __init__(self, random_state=42, **model_args):
        """
        :param model_args: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFRegressor
        """
        super().__init__(xgboost.XGBRFRegressor, random_state=random_state, **model_args)

    def is_sample_weight_supported(self) -> bool:
        return True


class XGBGradientBoostedVectorClassificationModel(AbstractSkLearnVectorClassificationModel, FeatureImportanceProviderSkLearnClassification):
    """
    XGBoost's classification model using gradient boosted trees
    """
    def __init__(self, random_state=42, use_balanced_class_weights=False, **model_args):
        """
        :param model_args: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
        """
        use_label_encoding = is_xgboost_version_at_least(1, 6)
        super().__init__(xgboost.XGBClassifier, random_state=random_state, use_balanced_class_weights=use_balanced_class_weights,
            use_label_encoding=use_label_encoding, **model_args)

    def is_sample_weight_supported(self) -> bool:
        return True


class XGBRandomForestVectorClassificationModel(AbstractSkLearnVectorClassificationModel, FeatureImportanceProviderSkLearnClassification):
    """
    XGBoost's random forest classification model
    """
    def __init__(self, random_state=42, use_balanced_class_weights=False, **model_args):
        """
        :param model_args: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFClassifier
        """
        use_label_encoding = is_xgboost_version_at_least(1, 6)
        super().__init__(xgboost.XGBRFClassifier, random_state=random_state, use_balanced_class_weights=use_balanced_class_weights,
            use_label_encoding=use_label_encoding, **model_args)

    def is_sample_weight_supported(self) -> bool:
        return True
