import io
import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple, Callable, Optional, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .torch_data import TensorScaler, VectorDataUtil, ClassificationVectorDataUtil, TorchDataSet, \
    TorchDataSetProviderFromDataUtil, TorchDataSetProvider, Tensoriser, TorchDataSetFromDataFrames
from .torch_enums import ClassificationOutputMode
from .torch_opt import NNOptimiser, NNLossEvaluatorRegression, NNLossEvaluatorClassification, NNOptimiserParams, TrainingInfo
from ..normalisation import NormalisationMode
from ..util.dtype import toFloatArray
from ..util.string import objectRepr, ToStringMixin
from ..vector_model import VectorRegressionModel, VectorClassificationModel

log: logging.Logger = logging.getLogger(__name__)


class MCDropoutCapableNNModule(nn.Module, ABC):
    """
    Base class for NN modules that are to support MC-Dropout.
    Support can be added by applying the _dropout function in the module's forward method.
    Then, to apply inference that samples results, call inferMCDropout rather than just using __call__.
    """

    def __init__(self) -> None:
        super().__init__()
        self._applyMCDropout = False
        self._pMCDropoutOverride = None

    def __setstate__(self, d: dict) -> None:
        if "_applyMCDropout" not in d:
            d["_applyMCDropout"] = False
        if "_pMCDropoutOverride" not in d:
            d["_pMCDropoutOverride"] = None
        super().__setstate__(d)

    def _dropout(self, x: torch.Tensor, pTraining=None, pInference=None) -> torch.Tensor:
        """
        This method is to to applied within the module's forward method to apply dropouts during training and/or inference.

        :param x: the model input tensor
        :param pTraining: the probability with which to apply dropouts during training; if None, apply no dropout
        :param pInference:  the probability with which to apply dropouts during MC-Dropout-based inference (via inferMCDropout,
            which may override the probability via its optional argument);
            if None, a dropout is not to be applied
        :return: a potentially modified version of x with some elements dropped out, depending on application context and dropout probabilities
        """
        if self.training and pTraining is not None:
            return F.dropout(x, pTraining)
        elif not self.training and self._applyMCDropout and pInference is not None:
            return F.dropout(x, pInference if self._pMCDropoutOverride is None else self._pMCDropoutOverride)
        else:
            return x

    def _enableMCDropout(self, enabled=True, pMCDropoutOverride=None) -> None:
        self._applyMCDropout = enabled
        self._pMCDropoutOverride = pMCDropoutOverride

    def inferMCDropout(self, x: Union[torch.Tensor, Sequence[torch.Tensor]], numSamples, p=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies inference using MC-Dropout, drawing the given number of samples.

        :param x: the model input (a tensor or tuple/list of tensors)
        :param numSamples: the number of samples to draw with MC-Dropout
        :param p: the dropout probability to apply, overriding the probability specified by the model's forward method; if None, use model's default
        :return: a pair (y, sd) where y the mean output tensor and sd is a tensor of the same dimension containing standard deviations
        """
        if type(x) not in (tuple, list):
            x = [x]
        results = []
        self._enableMCDropout(True, pMCDropoutOverride=p)
        try:
            for i in range(numSamples):
                y = self(*x)
                results.append(y)
        finally:
            self._enableMCDropout(False)
        results = torch.stack(results)
        mean = torch.mean(results, 0)
        stddev = torch.std(results, 0, unbiased=False)
        return mean, stddev


class TorchModel(ABC, ToStringMixin):
    """
    sensAI abstraction for torch models, which supports one-line training, allows for convenient model application,
    has basic mechanisms for data scaling, and soundly handles persistence (via pickle).
    An instance wraps a torch.nn.Module, which is constructed on demand during training via the factory method
    createTorchModule.
    """
    log: logging.Logger = log.getChild(__qualname__)

    def __init__(self, cuda=True) -> None:
        self.cuda: bool = cuda
        self.module: Optional[torch.nn.Module] = None
        self.outputScaler: Optional[TensorScaler] = None
        self.inputScaler: Optional[TensorScaler] = None
        self.trainingInfo: Optional[TrainingInfo] = None
        self._gpu: Optional[int] = None

    def setTorchModule(self, module: torch.nn.Module) -> None:
        self.module = module

    def getModuleBytes(self) -> bytes:
        bytesIO = io.BytesIO()
        torch.save(self.module, bytesIO)
        return bytesIO.getvalue()

    def setModuleBytes(self, modelBytes: bytes) -> None:
        modelFile = io.BytesIO(modelBytes)
        self._loadModel(modelFile)

    def getTorchModule(self) -> torch.nn.Module:
        return self.module

    def _setCudaEnabled(self, isCudaEnabled: bool) -> None:
        self.cuda = isCudaEnabled

    def _isCudaEnabled(self) -> bool:
        return self.cuda

    def _loadModel(self, modelFile) -> None:  # TODO: complete type hints: what types are allowed for modelFile?
        try:
            self.module = torch.load(modelFile)
            self._gpu = self._getGPUFromModelParameterDevice()
        except:
            if self._isCudaEnabled():
                if torch.cuda.device_count() > 0:
                    newDevice = "cuda:0"
                else:
                    newDevice = "cpu"
                self.log.warning(f"Loading of CUDA model failed, trying to map model to device {newDevice}...")
                if type(modelFile) != str:
                    modelFile.seek(0)
                try:
                    self.module = torch.load(modelFile, map_location=newDevice)
                except:
                    self.log.warning(f"Failure to map model to device {newDevice}, trying CPU...")
                    if newDevice != "cpu":
                        newDevice = "cpu"
                        self.module = torch.load(modelFile, map_location=newDevice)
                if newDevice == "cpu":
                    self._setCudaEnabled(False)
                    self._gpu = None
                else:
                    self._gpu = 0
                self.log.info(f"Model successfully loaded to {newDevice}")
            else:
                raise

    @abstractmethod
    def createTorchModule(self) -> torch.nn.Module:
        pass

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        del state["module"]
        state["modelBytes"] = self.getModuleBytes()
        return state

    def __setstate__(self, d: dict) -> None:
        # backward compatibility
        if "bestEpoch" in d:
            d["trainingInfo"] = TrainingInfo(bestEpoch=d["bestEpoch"])
            del d["bestEpoch"]

        modelBytes = None
        if "modelBytes" in d:
            modelBytes = d["modelBytes"]
            del d["modelBytes"]
        self.__dict__ = d
        if modelBytes is not None:
            self.setModuleBytes(modelBytes)

    def apply(self, X: Union[torch.Tensor, np.ndarray, TorchDataSet, Sequence[torch.Tensor]], asNumpy: bool = True, createBatch: bool = False,
            mcDropoutSamples: Optional[int] = None, mcDropoutProbability: Optional[float] = None, scaleOutput: bool = False,
            scaleInput: bool = False) -> Union[torch.Tensor, np.ndarray, Tuple]:
        """
        Applies the model to the given input tensor and returns the result (normalized)

        :param X: the input tensor (either a batch or, if createBatch=True, a single data point), a data set or a tuple/list of tensors
            (if the model accepts more than one input).
            If it is a data set, it will be processed at once, so the data set must not be too large to be processed at once.
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

        model = self.getTorchModule()
        model.eval()

        if isinstance(X, TorchDataSet):
            X = next(X.iterBatches(X.size(), inputOnly=True, shuffle=False))
        elif isinstance(X, np.ndarray):
            X = toFloatArray(X)
            X = torch.from_numpy(X).float()

        if type(X) not in (list, tuple):
            inputs = [X]
        else:
            inputs = X

        if self._isCudaEnabled():
            torch.cuda.set_device(self._gpu)
            inputs = [t.cuda() for t in inputs]
        if scaleInput:
            inputs = [self.inputScaler.normalise(t) for t in inputs]
        if createBatch:
            inputs = [t.view(1, *X.size()) for t in inputs]

        maxValue = max([t.max().item() for t in inputs])
        if maxValue > 2:
            log.warning("Received input which is likely to not be correctly normalised: maximum value in input tensor is %f" % maxValue)

        if mcDropoutSamples is None:
            y = model(*inputs)
            return extract(y)
        else:
            y, stddev = model.inferMCDropout(X, mcDropoutSamples, p=mcDropoutProbability)
            return extract(y), extract(stddev)

    def applyScaled(self, X: Union[torch.Tensor, np.ndarray, TorchDataSet, Sequence[torch.Tensor]], **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """
        applies the model to the given input tensor and returns the scaled result (i.e. in the original scale)

        :param X: the input tensor(s) or data set
        :param kwargs: parameters to pass on to apply

        :return: a scaled output tensor or, if MC-Dropout is applied, a pair (y, sd) of scaled tensors, where
            y the mean output tensor and sd is a tensor of the same dimension containing standard deviations
        """
        return self.apply(X, scaleOutput=True, scaleInput=True, **kwargs)

    def scaledOutput(self, output: torch.Tensor) -> torch.Tensor:
        return self.outputScaler.denormalise(output)

    def _extractParamsFromData(self, data: TorchDataSetProvider) -> None:
        self.outputScaler = data.getOutputTensorScaler()
        self.inputScaler = data.getInputTensorScaler()

    def fit(self, data: TorchDataSetProvider, nnOptimiserParams: NNOptimiserParams, strategy: "TorchModelFittingStrategy" = None) -> None:
        """
        Fits this model using the given model and strategy

        :param data: a provider for the data with which to fit the model
        :param strategy: the fitting strategy; if None, use TorchModelFittingStrategyDefault.
            Pass your own strategy to perform custom fitting processes, e.g. process which involve multi-stage learning
        :param nnOptimiserParams: the parameters with which to create an optimiser which can be applied in the fitting strategy
        """
        self._extractParamsFromData(data)
        optimiser = NNOptimiser(nnOptimiserParams)
        if strategy is None:
            strategy = TorchModelFittingStrategyDefault()
        self.trainingInfo = strategy.fit(self, data, optimiser)
        self._gpu = self._getGPUFromModelParameterDevice()

    def _getGPUFromModelParameterDevice(self) -> Optional[int]:
        try:
            return next(self.module.parameters()).get_device()
        except:
            return None

    @property
    def bestEpoch(self) -> Optional[int]:
        return self.trainingInfo.bestEpoch if self.trainingInfo is not None else None

    @property
    def totalEpochs(self) -> Optional[int]:
        return self.trainingInfo.totalEpochs if self.trainingInfo is not None else None

    def _toStringExcludes(self) -> List[str]:
        return ['_gpu', 'module', 'trainingInfo', "inputScaler", "outputScaler"]

    def _toStringAdditionalEntries(self):
        return dict(bestEpoch=self.bestEpoch, totalEpochs=self.totalEpochs)


class TorchModelFittingStrategy(ABC):
    """
    Defines the interface for fitting strategies that can be used in TorchModel.fit
    """
    @abstractmethod
    def fit(self, model: TorchModel, data: TorchDataSetProvider, nnOptimiser: NNOptimiser) -> Optional[TrainingInfo]:
        pass


class TorchModelFittingStrategyDefault(TorchModelFittingStrategy):
    """
    Represents the default fitting strategy, which simply applies the given optimiser to the model and data
    """
    def fit(self, model: TorchModel, data: TorchDataSetProvider, nnOptimiser: NNOptimiser) -> Optional[TrainingInfo]:
        return nnOptimiser.fit(model, data)


class TorchModelFromModuleFactory(TorchModel):
    def __init__(self, moduleFactory: Callable[..., torch.nn.Module], *args, cuda: bool = True, **kwargs) -> None:
        super().__init__(cuda)
        self.args = args
        self.kwargs = kwargs
        self.moduleFactory = moduleFactory

    def createTorchModule(self) -> torch.nn.Module:
        return self.moduleFactory(*self.args, **self.kwargs)


class VectorTorchModel(TorchModel, ABC):
    """
    Base class for TorchModels that can be used within VectorModels, where the input and output dimensions
    are determined by the data
    """
    def __init__(self, cuda: bool = True) -> None:
        super().__init__(cuda=cuda)
        self.inputDim = None
        self.outputDim = None

    def _extractParamsFromData(self, data: TorchDataSetProvider) -> None:
        super()._extractParamsFromData(data)
        self.inputDim = data.getInputDim()
        self.outputDim = data.getModelOutputDim()

    def createTorchModule(self) -> torch.nn.Module:
        return self.createTorchModuleForDims(self.inputDim, self.outputDim)

    @abstractmethod
    def createTorchModuleForDims(self, inputDim: int, outputDim: int) -> torch.nn.Module:
        pass


class TorchVectorRegressionModel(VectorRegressionModel):
    """
    Base class for the implementation of VectorRegressionModels based on TorchModels.
    An instance of this class will have an instance of TorchModel as the underlying model.
    """

    def __init__(self, modelClass: Callable[..., TorchModel], modelArgs: Sequence = (), modelKwArgs: Optional[dict] = None,
            normalisationMode: NormalisationMode = NormalisationMode.NONE,
            nnOptimiserParams: Union[dict, NNOptimiserParams, None] = None) -> None:
        """
        :param modelClass: the constructor with which to create the wrapped torch vector model
        :param modelArgs: the constructor argument list to pass to modelClass
        :param modelKwArgs: the dictionary of constructor keyword arguments to pass to modelClass
        :param normalisationMode: the normalisation mode to apply to input data frames
        :param nnOptimiserParams: the parameters to apply in NNOptimiser during training
        """
        super().__init__()
        if modelKwArgs is None:
            modelKwArgs = {}

        if nnOptimiserParams is None:
            nnOptimiserParamsInstance = NNOptimiserParams()
        else:
            nnOptimiserParamsInstance = NNOptimiserParams.fromDictOrInstance(nnOptimiserParams)
        if nnOptimiserParamsInstance.lossEvaluator is None:
            nnOptimiserParamsInstance.lossEvaluator = NNLossEvaluatorRegression(NNLossEvaluatorRegression.LossFunction.MSELOSS)

        self.normalisationMode = normalisationMode
        self.nnOptimiserParams = nnOptimiserParamsInstance
        self.modelClass = modelClass
        self.modelArgs = modelArgs
        self.modelKwArgs = modelKwArgs
        self.model: Optional[TorchModel] = None
        self.inputTensoriser = None

    def __setstate__(self, state) -> None:
        state["nnOptimiserParams"] = NNOptimiserParams.fromDictOrInstance(state["nnOptimiserParams"])
        if "inputTensoriser" not in state:
            state["inputTensoriser"] = None
        s = super()
        if hasattr(s, '__setstate__'):
            s.__setstate__(state)
        else:
            self.__dict__ = state

    def withInputTensoriser(self, tensoriser: Tensoriser) -> __qualname__:
        self.inputTensoriser = tensoriser
        return self

    def _createTorchModel(self) -> TorchModel:
        return self.modelClass(*self.modelArgs, **self.modelKwArgs)

    def _createDataSetProvider(self, inputs: pd.DataFrame, outputs: pd.DataFrame) -> TorchDataSetProvider:
        dataUtil = VectorDataUtil(inputs, outputs, self.model.cuda, normalisationMode=self.normalisationMode, inputTensoriser=self.inputTensoriser)
        return TorchDataSetProviderFromDataUtil(dataUtil, self.model.cuda)

    def _fit(self, inputs: pd.DataFrame, outputs: pd.DataFrame) -> None:
        self.model = self._createTorchModel()
        dataSetProvider = self._createDataSetProvider(inputs, outputs)
        self.model.fit(dataSetProvider, self.nnOptimiserParams)

    def _predictOutputsForInputDataFrame(self, inputs: pd.DataFrame) -> np.ndarray:
        batchSize = self.nnOptimiserParams.batchSize
        results = []
        dataSet = TorchDataSetFromDataFrames(inputs, None, self.model.cuda, inputTensoriser=self.inputTensoriser)
        for inputBatch in dataSet.iterBatches(batchSize, inputOnly=True):
            results.append(self.model.applyScaled(inputBatch, asNumpy=True))
        return np.concatenate(results)

    def _predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        yArray = self._predictOutputsForInputDataFrame(inputs)
        return pd.DataFrame(yArray, columns=self.getModelOutputVariableNames())

    def __str__(self) -> str:
        return objectRepr(self, ["model", "normalisationMode", "nnOptimiserParams"])


class TorchVectorClassificationModel(VectorClassificationModel):
    """
    Base class for the implementation of VectorClassificationModels based on TorchModels.
    An instance of this class will have an instance of TorchModel as the underlying model.
    """
    def __init__(self, outputMode: ClassificationOutputMode,
            modelClass: Callable[..., VectorTorchModel], modelArgs: Sequence = (), modelKwArgs: Optional[dict] = None,
            normalisationMode: NormalisationMode = NormalisationMode.NONE,
            nnOptimiserParams: Union[dict, NNOptimiserParams, None] = None) -> None:
        """
        :param outputMode: specifies the nature of the output of the underlying neural network model
        :param modelClass: the constructor with which to create the wrapped torch vector model
        :param modelArgs: the constructor argument list to pass to modelClass
        :param modelKwArgs: the dictionary of constructor keyword arguments to pass to modelClass
        :param normalisationMode: the normalisation mode to apply to input data frames
        :param nnOptimiserParams: the parameters to apply in NNOptimiser during training
        """
        super().__init__()
        if modelKwArgs is None:
            modelKwArgs = {}

        if nnOptimiserParams is None:
            nnOptimiserParamsInstance = NNOptimiserParams()
        else:
            nnOptimiserParamsInstance = NNOptimiserParams.fromDictOrInstance(nnOptimiserParams)
        if nnOptimiserParamsInstance.lossEvaluator is None:
            lossFunction = NNLossEvaluatorClassification.LossFunction.defaultForOutputMode(outputMode)
            nnOptimiserParamsInstance.lossEvaluator = NNLossEvaluatorClassification(lossFunction)

        self.outputMode = outputMode
        self.normalisationMode = normalisationMode
        self.nnOptimiserParams = nnOptimiserParams
        self.modelClass = modelClass
        self.modelArgs = modelArgs
        self.modelKwArgs = modelKwArgs
        self.model: Optional[VectorTorchModel] = None
        self.inputTensoriser = None

    def __setstate__(self, state) -> None:
        state["nnOptimiserParams"] = NNOptimiserParams.fromDictOrInstance(state["nnOptimiserParams"])
        if "inputTensoriser" not in state:
            state["inputTensoriser"] = None
        if "outputMode" not in state:
            state["outputMode"] = ClassificationOutputMode.PROBABILITIES
        s = super()
        if hasattr(s, '__setstate__'):
            s.__setstate__(state)
        else:
            self.__dict__ = state

    def withInputTensoriser(self, tensoriser: Tensoriser) -> __qualname__:
        self.inputTensoriser = tensoriser
        return self

    def _createTorchModel(self) -> VectorTorchModel:
        return self.modelClass(*self.modelArgs, **self.modelKwArgs)

    def _createDataSetProvider(self, inputs: pd.DataFrame, outputs: pd.DataFrame) -> TorchDataSetProvider:
        dataUtil = ClassificationVectorDataUtil(inputs, outputs, self.model.cuda, len(self._labels),
            normalisationMode=self.normalisationMode, inputTensoriser=self.inputTensoriser)
        return TorchDataSetProviderFromDataUtil(dataUtil, self.model.cuda)

    def _fitClassifier(self, inputs: pd.DataFrame, outputs: pd.DataFrame) -> None:
        if len(outputs.columns) != 1:
            raise ValueError("Expected one output dimension: the class labels")

        # transform outputs: for each data point, the new output shall be the index in the list of labels
        labels: pd.Series = outputs.iloc[:, 0]
        outputs = pd.DataFrame([self._labels.index(l) for l in labels], columns=outputs.columns, index=outputs.index)

        self.model = self._createTorchModel()

        dataSetProvider = self._createDataSetProvider(inputs, outputs)
        self.model.fit(dataSetProvider, self.nnOptimiserParams)

    def _predictOutputsForInputDataFrame(self, inputs: pd.DataFrame) -> torch.Tensor:
        batchSize = self.nnOptimiserParams.batchSize
        results = []
        dataSet = TorchDataSetFromDataFrames(inputs, None, self.model.cuda, inputTensoriser=self.inputTensoriser)
        for inputBatch in dataSet.iterBatches(batchSize, inputOnly=True):
            results.append(self.model.applyScaled(inputBatch, asNumpy=False))
        return torch.cat(results, dim=0)

    def _predictClassProbabilities(self, inputs: pd.DataFrame) -> pd.DataFrame:
        y = self._predictOutputsForInputDataFrame(inputs)
        if self.outputMode == ClassificationOutputMode.PROBABILITIES:
            pass
        elif self.outputMode == ClassificationOutputMode.LOG_PROBABILITIES:
            y = y.exp()
        elif self.outputMode == ClassificationOutputMode.UNNORMALISED_LOG_PROBABILITIES:
            y = y.softmax(dim=1)
        else:
            raise ValueError(f"Unhandled output mode {self.outputMode}")
        return pd.DataFrame(y.numpy(), columns=self._labels)

    def __str__(self) -> str:
        return objectRepr(self, ["model", "normalisationMode", "nnOptimiserParams"])
