import logging
import os.path
from typing import Sequence, Optional

import pandas as pd
from autogluon.tabular import TabularPredictor

from sensai import VectorRegressionModel
from sensai.util.pickle import getstate, setstate


def _fix_autogluon_logging():
    ag_logger = logging.getLogger("autogluon")
    ag_logger.propagate = True
    ag_logger.handlers = []


_fix_autogluon_logging()


class AutoGluonVectorRegressionModel(VectorRegressionModel):
    def __init__(
        self,
        models_dir: str,
        check_input_columns: bool = True,
        presets: Optional[str] = "medium_quality",
        time_limit: Optional[float] = None,
        excluded_model_types: Optional[Sequence[str]] = ("KNN", "RF", "XT"),
    ):
        """
        :param models_path: path to a directory in which all models shall be stored
        :param check_input_columns:
        :param presets:
        :param time_limit:
        :param excluded_model_types:
            It is very hard to find which values are allowed here. The only reference I found was
            https://auto.gluon.ai/0.1.0/tutorials/tabular_prediction/tabular-faq.html?highlight=excluded_model_types#how-can-i-skip-some-particular-models
            Also, NN is valid.
        """
        self.model: TabularPredictor | None = None
        self.models_dir = os.path.relpath(models_dir)
        self.presets = presets
        self.time_limit = time_limit
        self.excluded_model_types = excluded_model_types
        super().__init__(check_input_columns=check_input_columns)

    def __getstate__(self):
        return getstate(AutoGluonVectorRegressionModel, self, excluded_properties=["model"])

    def __setstate__(self, state):
        state["models_dir"] = state["models_dir"].replace("/", os.path.sep).replace("\\", os.path.sep)
        state["model"] = TabularPredictor.load(state["models_dir"])
        setstate(AutoGluonVectorRegressionModel, self, state)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame | list:
        target_label = self.model.label
        predictions = self.model.predict(x).values
        return pd.DataFrame({target_label: predictions}, index=x.index)

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame, weights: pd.DataFrame | None = None):
        if len(y.columns) != 1:
            raise ValueError(f"{self.__class__.__name__} currently only supports single target regression")
        target_name = y.columns[0]
        train_data = pd.concat([x, y], axis=1)
        self.model = TabularPredictor(
            label=target_name,
            problem_type="regression",
            eval_metric="r2",
            path=None)
        self.model.fit(
            train_data,
            presets=self.presets,
            time_limit=self.time_limit,
            excluded_model_types=self.excluded_model_types)
        self.model = self.model.clone_for_deployment(self.models_dir, return_clone=True, dirs_exist_ok=True)
