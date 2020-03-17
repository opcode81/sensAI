import io
import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple, Callable, Optional

import numpy as np
import pandas as pd
import torch

from ..util.dtype import toFloatArray
from ..vector_model import VectorRegressionModel, VectorClassificationModel
from ..util.string import objectRepr
from .torch_data import TensorScaler, VectorDataUtil, ClassificationVectorDataUtil, TorchDataSet, \
    TorchDataSetProviderFromDataUtil, TorchDataSetProvider
from .torch_opt import NNOptimiser, NNLossEvaluatorRegression, NNLossEvaluatorClassification

log = logging.getLogger(__name__)


class WrappedTorchModule(ABC):
    log = log.getChild(__qualname__)

    def __init__(self, cuda=True):
        self.cuda = cuda
        self.model = None
        self.outputScaler: Optional[TensorScaler] = None
        self.inputScaler: Optional[TensorScaler] = None

    def setTorchModel(self, model):
        self.model = model

    def getModelBytes(self):
        bytesIO = io.BytesIO()
        torch.save(self.model, bytesIO)
        return bytesIO.getvalue()

    def setModelBytes(self, modelBytes):
        modelFile = io.BytesIO(modelBytes)
        self._loadModel(modelFile)

    def getTorchModel(self):
        return self.model

    def _setCudaEnabled(self, isCudaEnabled):
        self.cuda = isCudaEnabled

    def _isCudaEnabled(self):
        return self.cuda

    def _loadModel(self, modelFile):
        try:
            self.model = torch.load(modelFile)
        except:
            if self._isCudaEnabled():
                self.log.warning("Loading of CUDA model failed, trying without CUDA...")
                if type(modelFile) != str:
                    modelFile.seek(0)
                self.model = torch.load(modelFile, map_location='cpu')
                self._setCudaEnabled(False)
                self.log.info("Model successfully loaded to CPU")
            else:
                raise

    @abstractmethod
    def createTorchModule(self):
        pass

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["model"]
        state["modelBytes"] = self.getModelBytes()
        return state

    def __setstate__(self, d):
        modelBytes = None
        if "modelBytes" in d:
            modelBytes = d["modelBytes"]
            del d["modelBytes"]
        self.__dict__ = d
        if modelBytes is not None:
            self.setModelBytes(modelBytes)

    def apply(self, X: Union[torch.Tensor, np.ndarray, TorchDataSet], asNumpy=True, createBatch=False, mcDropoutSamples=None, mcDropoutProbability=None, scaleOutput=False,
            scaleInput=False) -> Union[torch.Tensor, np.ndarray, Tuple]:
        """
        Applies the model to the given input tensor and returns the result (normalized)

        :param X: the input tensor (either a batch or, if createBatch=True, a single data point) or data set.
            If it is a data set, a single tensor will be extracted from it, so the data set must not be too large to be processed at once.
        :param asNumpy: flag indicating whether to convert the result to a numpy.array (if False, return tensor)
        :param createBatch: whether to add an additional tensor dimension for a batch containing just one data point
        :param mcDropoutSamples: if not None, apply MC-Dropout-based inference with the respective number of samples; if None, apply regular inference
        :param mcDropoutProbability: the probability with which to apply dropouts in MC-Dropout-based inference; if None, use model's default
        :param scaleOutput: whether to scale the output that is produced by the underlying model (using this instance's output scaler)
        :param scaleInput: whether to scale the input (using this instance's input scaler) before applying the underlying model

        :return: an output tensor or, if MC-Dropout is applied, a pair (y, sd) where y the mean output tensor and sd is a tensor of the same dimension
            containing standard deviations
        """
        def extract(z):
            if scaleOutput:
                z = self.scaledOutput(z)
            if self._isCudaEnabled():
                z = z.cpu()
            z = z.detach()
            if asNumpy:
                z = z.numpy()
            return z

        model = self.getTorchModel()
        model.eval()

        if isinstance(X, TorchDataSet):
            X = next(X.iterBatches(X.size(), inputOnly=True, shuffle=False))
        elif isinstance(X, np.ndarray):
            X = toFloatArray(X)
            X = torch.from_numpy(X).float()

        if self._isCudaEnabled():
            X = X.cuda()
        if scaleInput:
            X = self.inputScaler.normalise(X)
        if createBatch:
            X = X.view(1, *X.size())

        maxValue = X.max().item()
        if maxValue > 2:
            log.warning("Received input which is likely to not be correctly normalised: maximum value in input tensor is %f" % maxValue)

        if mcDropoutSamples is None:
            y = model(X)
            return extract(y)
        else:
            y, stddev = model.inferMCDropout(X, mcDropoutSamples, p=mcDropoutProbability)
            return extract(y), extract(stddev)

    def applyScaled(self, X: Union[torch.Tensor, np.ndarray, TorchDataSet], **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """
        applies the model to the given input tensor and returns the scaled result (i.e. in the original scale)

        :param X: the input tensor or data set
        :param kwargs: parameters to pass on to apply

        :return: a scaled output tensor or, if MC-Dropout is applied, a pair (y, sd) of scaled tensors, where
            y the mean output tensor and sd is a tensor of the same dimension containing standard deviations
        """
        return self.apply(X, scaleOutput=True, scaleInput=True, **kwargs)

    def scaledOutput(self, output):
        return self.outputScaler.denormalise(output)

    def _extractParamsFromData(self, data: TorchDataSetProvider):
        self.outputScaler = data.getOutputTensorScaler()
        self.inputScaler = data.getInputTensorScaler()

    def fit(self, data: TorchDataSetProvider, **nnOptimiserParams):
        self._extractParamsFromData(data)
        optimiser = NNOptimiser(cuda=self.cuda, **nnOptimiserParams)
        optimiser.fit(self, data)


class WrappedTorchVectorModule(WrappedTorchModule, ABC):
    """
    Base class for wrapped torch modules that map vectors to vectors
    """
    def __init__(self, cuda: bool = True):
        super().__init__(cuda=cuda)
        self.inputDim = None
        self.outputDim = None

    def _extractParamsFromData(self, data: TorchDataSetProvider):
        super()._extractParamsFromData(data)
        self.inputDim = data.getInputDim()
        self.outputDim = data.getModelOutputDim()

    def createTorchModule(self):
        return self.createTorchVectorModule(self.inputDim, self.outputDim)

    @abstractmethod
    def createTorchVectorModule(self, inputDim, outputDim):
        pass


class TorchVectorRegressionModel(VectorRegressionModel):
    def __init__(self, modelClass: Callable[..., WrappedTorchVectorModule], modelArgs, modelKwArgs, normalisationMode, nnOptimiserParams):
        """
        :param modelClass:
        :param modelArgs:
        :param modelKwArgs:
        :param normalisationMode:
        :param nnOptimiserParams:
        """
        super().__init__()
        if "lossEvaluator" not in nnOptimiserParams:
            nnOptimiserParams["lossEvaluator"] = NNLossEvaluatorRegression(NNLossEvaluatorRegression.LossFunction.MSELOSS)
        self.normalisationMode = normalisationMode
        self.nnOptimiserParams = nnOptimiserParams
        self.modelClass = modelClass
        self.modelArgs = modelArgs
        self.modelKwArgs = modelKwArgs
        self.model: Optional[WrappedTorchVectorModule] = None

    def createTorchVectorModel(self) -> WrappedTorchVectorModule:
        return self.modelClass(*self.modelArgs, **self.modelKwArgs)

    def _fit(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        self.model = self.createTorchVectorModel()
        dataUtil = VectorDataUtil(inputs, outputs, self.model.cuda, normalisationMode=self.normalisationMode)
        dataSetProvider = TorchDataSetProviderFromDataUtil(dataUtil, self.model.cuda)
        self.model.fit(dataSetProvider, **self.nnOptimiserParams)

    def _predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        yArray = self.model.applyScaled(inputs.values)
        return pd.DataFrame(yArray, columns=self.getModelOutputVariableNames())

    def __str__(self):
        return objectRepr(self, ["model", "normalisationMode", "nnOptimiserParams"])


class TorchVectorClassificationModel(VectorClassificationModel):
    def __init__(self, modelClass: Callable[..., WrappedTorchVectorModule], modelArgs, modelKwArgs, normalisationMode, nnOptimiserParams):
        """
        :param modelClass:
        :param modelArgs:
        :param modelKwArgs:
        :param normalisationMode:
        :param nnOptimiserParams:
        """
        super().__init__()
        if "lossEvaluator" not in nnOptimiserParams:
            nnOptimiserParams["lossEvaluator"] = NNLossEvaluatorClassification(NNLossEvaluatorClassification.LossFunction.CROSSENTROPY)
        self.normalisationMode = normalisationMode
        self.nnOptimiserParams = nnOptimiserParams
        self.modelClass = modelClass
        self.modelArgs = modelArgs
        self.modelKwArgs = modelKwArgs
        self.model: Optional[WrappedTorchVectorModule] = None

    def createTorchVectorModel(self) -> WrappedTorchVectorModule:
        return self.modelClass(*self.modelArgs, **self.modelKwArgs)

    def _createDataSetProvider(self, inputs: pd.DataFrame, outputs: pd.DataFrame) -> TorchDataSetProvider:
        dataUtil = ClassificationVectorDataUtil(inputs, outputs, self.model.cuda, len(self._labels),
            normalisationMode=self.normalisationMode)
        return TorchDataSetProviderFromDataUtil(dataUtil, self.model.cuda)

    def _fitClassifier(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        if len(outputs.columns) != 1:
            raise ValueError("Expected one output dimension: the class labels")

        # transform outputs: for each data point, the new output shall be the index in the list of labels
        labels: pd.Series = outputs.iloc[:, 0]
        outputs = pd.DataFrame([self._labels.index(l) for l in labels], columns=outputs.columns, index=outputs.index)

        self.model = self.createTorchVectorModel()

        dataSet = self._createDataSetProvider(inputs, outputs)
        self.model.fit(dataSet, **self.nnOptimiserParams)

    def _predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        return self.convertClassProbabilitiesToPredictions(self._predictClassProbabilities(inputs))

    def _predictOutputsForInputDataFrame(self, inputs: pd.DataFrame) -> np.ndarray:
        return self.model.applyScaled(inputs.values, asNumpy=True)

    def _predictClassProbabilities(self, inputs: pd.DataFrame):
        y = self._predictOutputsForInputDataFrame(inputs)
        normalisationConstants = y.sum(axis=1)
        for i in range(y.shape[0]):
            y[i,:] /= normalisationConstants[i]
        return pd.DataFrame(y, columns=self._labels)

    def __str__(self):
        return objectRepr(self, ["model", "normalisationMode", "nnOptimiserParams"])
