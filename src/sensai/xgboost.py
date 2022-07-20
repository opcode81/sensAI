import xgboost

from .sklearn.sklearn_base import AbstractSkLearnMultipleOneDimVectorRegressionModel, AbstractSkLearnVectorClassificationModel, \
    FeatureImportanceProviderSkLearnRegressionMultipleOneDim, FeatureImportanceProviderSkLearnClassification


class XGBGradientBoostedVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel, FeatureImportanceProviderSkLearnRegressionMultipleOneDim):
    """
    XGBoost's regression model using gradient boosted trees
    """
    def __init__(self, random_state=42, **modelArgs):
        """
        :param modelArgs: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor
        """
        super().__init__(xgboost.XGBRegressor, random_state=random_state, **modelArgs)


class XGBRandomForestVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel, FeatureImportanceProviderSkLearnRegressionMultipleOneDim):
    """
    XGBoost's random forest regression model
    """
    def __init__(self, random_state=42, **modelArgs):
        """
        :param modelArgs: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFRegressor
        """
        super().__init__(xgboost.XGBRFRegressor, random_state=random_state, **modelArgs)


class XGBGradientBoostedVectorClassificationModel(AbstractSkLearnVectorClassificationModel, FeatureImportanceProviderSkLearnClassification):
    """
    XGBoost's classification model using gradient boosted trees
    """
    def __init__(self, random_state=42, useBalancedClassWeights=False, **modelArgs):
        """
        :param modelArgs: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
        """
        super().__init__(xgboost.XGBClassifier, random_state=random_state, useBalancedClassWeights=useBalancedClassWeights, **modelArgs)


class XGBRandomForestVectorClassificationModel(AbstractSkLearnVectorClassificationModel, FeatureImportanceProviderSkLearnClassification):
    """
    XGBoost's random forest classification model
    """
    def __init__(self, random_state=42, useBalancedClassWeights=False, **modelArgs):
        """
        :param modelArgs: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFClassifier
        """
        super().__init__(xgboost.XGBRFClassifier, random_state=random_state, useBalancedClassWeights=useBalancedClassWeights, **modelArgs)
