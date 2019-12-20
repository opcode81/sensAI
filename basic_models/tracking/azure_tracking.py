from abc import ABC, abstractmethod
from typing import Union, Dict
from azureml.core import Experiment, Workspace, Run

from basic_models import VectorModel
from basic_models.evaluation import VectorModelEvaluator, VectorModelCrossValidator
from basic_models.hyperopt import GridSearch
from basic_models.tracking.tracking_base import TrackedExperiment, VectorModelEvaluatorTrackingWrapper, \
    VectorModelCrossValidationTrackingWrapper


class AzureMLExperiment(TrackedExperiment, ABC):
    """
    Class to automatically track parameters, metrics and artifacts for a single model with azureml-sdk
    """
    def __init__(self, experimentName: str, workspace: Workspace,
        evaluator: Union[VectorModelEvaluator, VectorModelCrossValidator, VectorModelEvaluatorTrackingWrapper,
        VectorModelCrossValidationTrackingWrapper]):
        """

        :param experimentName:
        :param workspace:
        :param evaluator:
        """
        super().__init__(experimentName=experimentName, evaluator=evaluator)
        self.experiment = Experiment(workspace=workspace, name=experimentName)

    @abstractmethod
    def apply(self, model: VectorModel):
        pass


class AzureMLSingleModelExperiment(AzureMLExperiment):
    """
    Class to automatically track parameters, metrics and artifacts for single models with azureml-sdk
    """
    def __init__(self, experimentName: str, workspace: Workspace,
        evaluator: Union[VectorModelEvaluatorTrackingWrapper, VectorModelCrossValidationTrackingWrapper]):
        """

        :param experimentName:
        :param workspace:
        :param evaluator:
        """
        super().__init__(experimentName=experimentName, workspace=workspace, evaluator=evaluator)

    def apply(self, model: VectorModel, additionalLoggingValuesDict: dict = None):
        with self.experiment.start_logging() as run:
            run.log('str(model)', model.__str__())
            valuesDict = self.evaluator.evaluate(model)
            if additionalLoggingValuesDict is not None:
                valuesDict.update(additionalLoggingValuesDict)
            for name, value in valuesDict.items():
                run.log(name, value)


class AzureMLGridSearchExperiment(AzureMLExperiment):
    """
    Class to automatically track parameters, metrics and artifacts for a grid search with azureml-sdk
    """
    def __init__(self, experimentName: str, workspace: Workspace,
        evaluator: Union[VectorModelEvaluator, VectorModelCrossValidator]):
        """

        :param experimentName:
        :param workspace:
        :param evaluator:
        """
        super().__init__(experimentName=experimentName, workspace=workspace, evaluator=evaluator)

    def _loggingCallback(self, valuesDict: Dict, additionalValuesDict: dict = None, parentRun: Run = None):
        with self.experiment.start_logging() if parentRun is None else parentRun.child_run() as run:
            if additionalValuesDict is not None:
                valuesDict.update(additionalValuesDict)
            for name, value in valuesDict.items():
                run.log(name, value)

    def apply(self, gridSearch: GridSearch, additionalValuesDict: dict = None, useChildRuns: bool = False):
        if not useChildRuns:
            gridSearch.run(evaluatorOrValidator=self.evaluator,
            loggingCallback=lambda x: self._loggingCallback(valuesDict=x, additionalValuesDict=additionalValuesDict))
        else:
            with self.experiment.start_logging() as parentRun:
                if additionalValuesDict is not None:
                    for name, value in additionalValuesDict.items():
                        parentRun.log(name, value)
                gridSearch.run(evaluatorOrValidator=self.evaluator,
                loggingCallback=lambda x: self._loggingCallback(valuesDict=x, parentRun=parentRun))

