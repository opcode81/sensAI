from typing import Dict

import xgboost

from .feature_importance import FeatureImportanceProvider
from .sklearn.sklearn_base import AbstractSkLearnMultipleOneDimVectorRegressionModel, AbstractSkLearnVectorClassificationModel


class XGBGradientBoostedVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel, FeatureImportanceProvider):
    """
    XGBoost's regression model using gradient boosted trees
    """
    def __init__(self, random_state=42, **modelArgs):
        """
        :param modelArgs: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor
        """
        super().__init__(xgboost.XGBRegressor, random_state=random_state, **modelArgs)

    def getFeatureImportanceDict(self) -> Dict[str, Dict[str, float]]:
        return {targetFeature: dict(zip(self._modelInputVariableNames, model.feature_importances_)) for targetFeature, model in self.models.items()}


class XGBRandomForestVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel, FeatureImportanceProvider):
    """
    XGBoost's random forest regression model
    """
    def __init__(self, random_state=42, **modelArgs):
        """
        :param modelArgs: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFRegressor
        """
        super().__init__(xgboost.XGBRFRegressor, random_state=random_state, **modelArgs)

    def getFeatureImportanceDict(self) -> Dict[str, Dict[str, float]]:
        return {targetFeature: dict(zip(self._modelInputVariableNames, model.feature_importances_)) for targetFeature, model in self.models.items()}


class XGBGradientBoostedVectorClassificationModel(AbstractSkLearnVectorClassificationModel, FeatureImportanceProvider):
    """
    XGBoost's classification model using gradient boosted trees
    """
    def __init__(self, random_state=42, **modelArgs):
        """
        :param modelArgs: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
        """
        super().__init__(xgboost.XGBClassifier, random_state=random_state, **modelArgs)

    def getFeatureImportanceDict(self) -> Dict[str, float]:
        return dict(zip(self._modelInputVariableNames, self.model.feature_importances_))


class XGBRandomForestVectorClassificationModel(AbstractSkLearnVectorClassificationModel, FeatureImportanceProvider):
    """
    XGBoost's random forest classification model
    """
    def __init__(self, random_state=42, **modelArgs):
        """
        :param modelArgs: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFClassifier
        """
        super().__init__(xgboost.XGBRFClassifier, random_state=random_state, **modelArgs)

    def getFeatureImportanceDict(self) -> Dict[str, float]:
        return dict(zip(self._modelInputVariableNames, self.model.feature_importances_))
