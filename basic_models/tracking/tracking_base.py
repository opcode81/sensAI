from abc import ABC, abstractmethod
from typing import Union

from basic_models import VectorRegressionModelEvaluator, VectorClassificationModelEvaluator, VectorModel, \
    VectorRegressionModelCrossValidator, VectorClassificationModelCrossValidator
from basic_models.basic_models_base import PredictorModel

from basic_models.evaluation import VectorModelEvaluator, VectorModelCrossValidator
from basic_models.hyperopt import GridSearch


class AbstractVectorModelEvaluatorTrackingWrapper(ABC):

    @abstractmethod
    def evaluate(self, model: PredictorModel):
        pass


class VectorModelEvaluatorTrackingWrapper(AbstractVectorModelEvaluatorTrackingWrapper):
    def __init__(self, vectorModelEvaluator: Union[VectorRegressionModelEvaluator, VectorClassificationModelEvaluator]):
        """
        Wrapper class for evaluators
        :param vectorModelEvaluator:
        """
        self.vectorModelEvaluator = vectorModelEvaluator

    def evaluate(self, vectorModel: VectorModel):
        self.vectorModelEvaluator.fitModel(vectorModel)
        evalData = self.vectorModelEvaluator.evalModel(vectorModel)
        return evalData.getEvalStats().getAll()


class VectorModelCrossValidationTrackingWrapper(AbstractVectorModelEvaluatorTrackingWrapper):
    def __init__(self, vectorModelCrossValidator: Union[VectorRegressionModelCrossValidator, VectorClassificationModelCrossValidator]):
        """
        Wrapper class for crossValidators
        :param vectorModelCrossValidator:
        """
        self.vectorModelCrossValidator = vectorModelCrossValidator

    def evaluate(self, vectorModel: VectorModel):
        evalData = self.vectorModelCrossValidator.evalModel(vectorModel)
        return evalData.getEvalStatsCollection().aggStats()


class TrackedExperiment(ABC):
    def __init__(self, experimentName: str,
                 evaluator: Union[VectorModelEvaluator, VectorModelCrossValidator, VectorModelEvaluatorTrackingWrapper,
                                  VectorModelCrossValidationTrackingWrapper]):

        self.experimentName = experimentName
        self.evaluator = evaluator

    @abstractmethod
    def apply(self, models: Union[VectorModel, GridSearch]):
        pass
