import xgboost

from dcs.sensai.sklearn.sklearn_base import AbstractSkLearnMultipleOneDimVectorRegressionModel, AbstractSkLearnVectorClassificationModel


class XGBGradientBosstedVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    """
    XGBoost's regression model using gradient boosted trees
    """
    def __init__(self, random_state=42, **modelArgs):
        """
        :param modelArgs: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor
        """
        super().__init__(xgboost.XGBRegressor, random_state=random_state, **modelArgs)


class XGBRandomForestVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    """
    XGBoost's random forest regression model
    """
    def __init__(self, random_state=42, **modelArgs):
        """
        :param modelArgs: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFRegressor
        """
        super().__init__(xgboost.XGBRFRegressor, random_state=random_state, **modelArgs)


class XGBGradientBoostedVectorClassifictionModel(AbstractSkLearnVectorClassificationModel):
    """
    XGBoost's classification model using gradient boosted trees
    """
    def __init__(self, random_state=42, **modelArgs):
        """
        :param modelArgs: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
        """
        super().__init__(xgboost.XGBClassifier, random_state=random_state, **modelArgs)


class XGBRandomForestVectorClassifictionModel(AbstractSkLearnVectorClassificationModel):
    """
    XGBoost's random forest classification model
    """
    def __init__(self, random_state=42, **modelArgs):
        """
        :param modelArgs: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFClassifier
        """
        super().__init__(xgboost.XGBRFClassifier, random_state=random_state, **modelArgs)
