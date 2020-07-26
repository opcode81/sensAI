from abc import ABC, abstractmethod
from typing import Union

from .eval_stats import ModelEvaluationData
from ..models.base import FitterModel, PredictorModel


class ModelEvaluator(ABC):
    @abstractmethod
    def fitModel(self, model: FitterModel):
        """Fits the given model's parameters using this evaluator's data"""
        pass

    @abstractmethod
    def evalModel(self, model: Union[FitterModel, PredictorModel], **kwargs) -> ModelEvaluationData:
        """
        Evaluates the given model

        :param model: the model to evaluate
        :return: the evaluation result
        """
        pass
