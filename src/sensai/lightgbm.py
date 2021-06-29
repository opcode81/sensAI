from typing import Sequence, Union, Optional, Dict
import logging
import lightgbm
import pandas as pd
import re

from .util.string import orRegexGroup
from .sklearn.sklearn_base import AbstractSkLearnMultipleOneDimVectorRegressionModel, AbstractSkLearnVectorClassificationModel

log = logging.getLogger(__name__)


class LightGBMVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    log = log.getChild(__qualname__)

    def __init__(self, categoricalFeatureNames: Optional[Union[Sequence[str], str]] = None, random_state=42, num_leaves=31,
            max_depth=-1, n_estimators=100, min_child_samples=20, importance_type="gain", **modelArgs):
        """
        :param categoricalFeatureNames: sequence of feature names in the input data that are categorical.
            Columns that have dtype 'category' (as will be the case for categorical columns created via FeatureGenerators)
            need not be specified (should be inferred automatically).
            In general, passing categorical features is preferable to using one-hot encoding, for example.
        :param random_state: the random seed to use
        :param num_leaves: the maximum number of leaves in one tree (original lightgbm default is 31)
        :param max_depth: maximum tree depth for base learners, <=0 means no limit
        :param n_estimators: number of boosted trees to fit
        :param min_child_samples: minimum number of data needed in a child (leaf)
        :param importance_type: the type of feature importance to be set in the respective property of the wrapped model.
            If ‘split’, result contains numbers of times the feature is used in a model.
            If ‘gain’, result contains total gains of splits which use the feature.
        :param modelArgs: see https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
        """
        super().__init__(lightgbm.sklearn.LGBMRegressor, random_state=random_state, num_leaves=num_leaves, importance_type=importance_type,
            max_depth=max_depth, n_estimators=n_estimators, min_child_samples=min_child_samples,
            **modelArgs)

        if type(categoricalFeatureNames) == str:
            categoricalFeatureNameRegex = categoricalFeatureNames
        else:
            if categoricalFeatureNames is not None and len(categoricalFeatureNames) > 0:
                categoricalFeatureNameRegex = orRegexGroup(categoricalFeatureNames)
            else:
                categoricalFeatureNameRegex = None
        self._categoricalFeatureNameRegex: str = categoricalFeatureNameRegex

    def _updateModelArgs(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        if self._categoricalFeatureNameRegex is not None:
            cols = list(inputs.columns)
            categoricalFeatureNames = [col for col in cols if re.match(self._categoricalFeatureNameRegex, col)]
            colIndices = [cols.index(f) for f in categoricalFeatureNames]
            args = {"cat_column": colIndices}
            self.log.info(f"Updating model parameters with {args}")
            self.modelArgs.update(args)

    def getFeatureImportances(self) -> Dict[str, Dict[str, int]]:
        return {targetFeature: dict(zip(model.feature_name_, model.feature_importances_)) for targetFeature, model in self.models.items()}


class LightGBMVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    log = log.getChild(__qualname__)

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

        if type(categoricalFeatureNames) == str:
            categoricalFeatureNameRegex = categoricalFeatureNames
        else:
            if categoricalFeatureNames is not None and len(categoricalFeatureNames) > 0:
                categoricalFeatureNameRegex = orRegexGroup(categoricalFeatureNames)
            else:
                categoricalFeatureNameRegex = None
        self._categoricalFeatureNameRegex: str = categoricalFeatureNameRegex

    def _updateModelArgs(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        if self._categoricalFeatureNameRegex is not None:
            cols = list(inputs.columns)
            categoricalFeatureNames = [col for col in cols if re.match(self._categoricalFeatureNameRegex, col)]
            colIndices = [cols.index(f) for f in categoricalFeatureNames]
            args = {"cat_column": colIndices}
            self.log.info(f"Updating model parameters with {args}")
            self.modelArgs.update(args)

    def getFeatureImportances(self) -> Dict[str, Dict[str, int]]:
        return dict(zip(self.model.feature_name_, self.model.feature_importances_))
