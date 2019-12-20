from typing import Union

import mlflow

from basic_models import VectorModel
from basic_models.tracking.tracking_base import TrackedExperiment, VectorModelEvaluatorTrackingWrapper, \
    VectorModelCrossValidationTrackingWrapper


class MlFlowExperiment(TrackedExperiment):
    """

    """
    def __init__(self, experimentName: str, trackingUri: str, evaluator: Union[VectorModelEvaluatorTrackingWrapper,
                                                                        VectorModelCrossValidationTrackingWrapper]):
        """

        :param experimentName:
        :param trackingUri:
        :param evaluator:
        """
        super().__init__(experimentName=experimentName, evaluator=evaluator)

        mlflow.set_tracking_uri(trackingUri)
        mlflow.set_experiment(experiment_name=experimentName)

    def apply(self, model: VectorModel, **kwargs):
        with mlflow.start_run():
            metrics = self.evaluator.evaluate(model)
            mlflow.log_param('model', str(model))
            mlflow.log_metrics(metrics)