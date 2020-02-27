import logging

import lightgbm
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.tree

import pandas as pd

from .sklearn_base import AbstractSkLearnVectorClassificationModel


_log = logging.getLogger(__name__)


class SkLearnDecisionTreeVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, min_samples_leaf=8, random_state=42, **modelArgs):
        super().__init__(sklearn.tree.DecisionTreeClassifier,
            min_samples_leaf=min_samples_leaf, random_state=random_state, **modelArgs)


class SkLearnRandomForestVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, min_samples_leaf=8, random_state=42, **modelArgs):
        super().__init__(sklearn.ensemble.RandomForestClassifier,
            random_state=random_state, min_samples_leaf=min_samples_leaf, **modelArgs)


class SkLearnMLPVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, **modelArgs):
        super().__init__(sklearn.neural_network.MLPClassifier, **modelArgs)


class SkLearnMultinomialNBVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, **modelArgs):
        super().__init__(sklearn.naive_bayes.MultinomialNB, **modelArgs)


class SkLearnLightGBMVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
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
        self.categoricalFeatureNames = categoricalFeatureNames

    def _updateModelArgs(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        if self.categoricalFeatureNames is not None:
            cols = list(inputs.columns)
            colIndices = [cols.index(f) for f in self.categoricalFeatureNames]
            args = {"cat_column": colIndices}
            self.log.info(f"Updating model parameters with {args}")
            self.modelArgs.update(args)


class SkLearnSVCVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, random_state=42, **modelArgs):
        super().__init__(sklearn.svm.SVC, random_state=random_state, **modelArgs)
