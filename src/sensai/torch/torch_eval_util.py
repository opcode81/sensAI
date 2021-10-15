from typing import Union

from dcs.sensai.evaluation import RegressionEvaluationUtil
from dcs.sensai.evaluation.crossval import VectorModelCrossValidationData
from dcs.sensai.evaluation.eval_util import TEvalData, TCrossValData
from dcs.sensai.evaluation.evaluator import VectorModelEvaluationData
from dcs.sensai.torch import TorchVectorRegressionModel


class TorchVectorRegressionModelEvaluationUtil(RegressionEvaluationUtil):

    def _createPlots(self, data: Union[TEvalData, TCrossValData], resultCollector: RegressionEvaluationUtil.ResultCollector, subtitle=None):
        super()._createPlots(data, resultCollector, subtitle)
        if isinstance(data, VectorModelEvaluationData):
            self._addLossProgressionPlotIfTorchVectorRegressionModel(data.model, "loss-progression", resultCollector)
        elif isinstance(data, VectorModelCrossValidationData):
            if data.trainedModels is not None:
                for i, model in enumerate(data.trainedModels, start=1):
                    self._addLossProgressionPlotIfTorchVectorRegressionModel(model, "loss-progression-{i}", resultCollector)

    @staticmethod
    def _addLossProgressionPlotIfTorchVectorRegressionModel(model, plotName, resultCollector):
        if isinstance(model, TorchVectorRegressionModel):
            resultCollector.addFigure(plotName, model.model.trainingInfo.plotAll())
