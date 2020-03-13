from typing import Sequence
import logging
import lightgbm
import pandas as pd

from .sklearn.sklearn_base import AbstractSkLearnMultipleOneDimVectorRegressionModel, AbstractSkLearnVectorClassificationModel

_log = logging.getLogger(__name__)


class LightGBMVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    _log = _log.getChild(__qualname__)

    def __init__(self, categoricalFeatureNames: Sequence[str] = None, random_state=42, num_leaves=31, **modelArgs):
        """
        :param categoricalFeatureNames: sequence of feature names in the input data that are categorical.
            Columns that have dtype 'category' (as will be the case for categorical columns created via FeatureGenerators)
            need not be specified (should be inferred automatically).
            In general, passing categorical features is preferable to using one-hot encoding, for example.
        :param random_state: the random seed to use
        :param num_leaves: the maximum number of leaves in one tree (original lightgbm default is 31)
        :param modelArgs: see https://lightgbm.readthedocs.io/en/latest/Parameters.html
        """
        self.categoricalFeatureNames = categoricalFeatureNames
        super().__init__(lightgbm.sklearn.LGBMRegressor, random_state=random_state, num_leaves=num_leaves, **modelArgs)

    def _updateModelArgs(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        if self.categoricalFeatureNames is not None:
            cols = list(inputs.columns)
            colIndices = [cols.index(f) for f in self.categoricalFeatureNames]
            args = {"cat_column": colIndices}
            self._log.info(f"Updating model parameters with {args}")
            self.modelArgs.update(args)


class LightGBMVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    _log = _log.getChild(__qualname__)

    def __init__(self, categoricalFeatureNames: Sequence[str] = None, random_state=42, num_leaves=31, **modelArgs):
        """
        :param categoricalFeatureNames: sequence of feature names in the input data that are categorical
            Columns that have dtype 'category' (as will be the case for categorical columns created via FeatureGenerators)
            need not be specified (should be inferred automatically, but we have never actually tested this behaviour
            successfully for a classification model).
            In general, passing categorical features may be preferable to using one-hot encoding, for example.
        :param random_state: the random seed to use
        :param num_leaves: the maximum number of leaves in one tree (original lightgbm default is 31)
        :param modelArgs: see https://lightgbm.readthedocs.io/en/latest/Parameters.html
        """
        super().__init__(lightgbm.sklearn.LGBMClassifier, random_state=random_state, num_leaves=num_leaves, **modelArgs)
        self.categoricalFeatureNames = categoricalFeatureNames

    def _updateModelArgs(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        if self.categoricalFeatureNames is not None:
            cols = list(inputs.columns)
            colIndices = [cols.index(f) for f in self.categoricalFeatureNames]
            args = {"cat_column": colIndices}
            self._log.info(f"Updating model parameters with {args}")
            self.modelArgs.update(args)