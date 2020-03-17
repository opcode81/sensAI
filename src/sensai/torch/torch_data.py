from abc import ABC, abstractmethod
from typing import Tuple, Sequence, Generator, Optional, Union

from torch.autograd import Variable
import pandas as pd
import numpy as np
import torch

from .. import normalisation


def toTensor(d: Union[torch.Tensor, np.ndarray, list], cuda=False):
    if not isinstance(d, torch.Tensor):
        if isinstance(d, np.ndarray):
            d = torch.from_numpy(d)
        elif isinstance(d, list):
            d = torch.from_numpy(np.array(d))
        else:
            raise ValueError()
    if cuda:
        d.cuda()
    return d


class TensorScaler(ABC):
    @abstractmethod
    def cuda(self):
        """
        Makes this scaler's components use CUDA
        """
        pass

    @abstractmethod
    def normalise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies scaling/normalisation to the given tensor
        :param tensor: the tensor to scale/normalise
        :return: the scaled/normalised tensor
        """
        pass

    @abstractmethod
    def denormalise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse of method normalise to the given tensor
        :param tensor: the tensor to denormalise
        :return: the denormalised tensor
        """
        pass


class TensorScalerFromVectorDataScaler(TensorScaler):
    def __init__(self, vectorDataScaler: normalisation.VectorDataScaler, cuda: bool):
        self.scale = vectorDataScaler.scale
        if self.scale is not None:
            self.scale = torch.from_numpy(vectorDataScaler.scale).float()
        self.translate = vectorDataScaler.translate
        if self.translate is not None:
            self.translate = torch.from_numpy(vectorDataScaler.translate).float()
        if cuda:
            self.cuda()

    def cuda(self):
        if self.scale is not None:
            self.scale = self.scale.cuda()
        if self.translate is not None:
            self.translate = self.translate.cuda()

    def normalise(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.translate is not None:
            tensor -= self.translate
        if self.scale is not None:
            tensor /= self.scale
        return tensor

    def denormalise(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.scale is not None:
            tensor *= self.scale
        if self.translate is not None:
            tensor += self.translate
        return tensor


class TensorScalerIdentity(TensorScaler):
    def cuda(self):
        pass

    def normalise(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def denormalise(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class DataUtil(ABC):
    """Interface for DataUtil classes, which are used to process data for neural networks"""

    @abstractmethod
    def splitInputOutputPairs(self, fractionalSizeOfFirstSet) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Splits the data set

        :param fractionalSizeOfFirstSet: the desired fractional size in
        :return: a tuple (A, B) where A and B are tuples (in, out) with input and output data
        """
        pass

    @abstractmethod
    def getOutputTensorScaler(self) -> TensorScaler:
        """
        Gets the scaler with which to scale model outputs

        :return: the scaler
        """
        pass

    @abstractmethod
    def getInputTensorScaler(self) -> TensorScaler:
        """
        Gets the scaler with which to scale model inputs

        :return: the scaler
        """
        pass

    @abstractmethod
    def modelOutputDim(self) -> int:
        """
        :return: the dimensionality that is to be output by the model to be trained
        """
        pass

    @abstractmethod
    def inputDim(self):
        """
        :return: the dimensionality of the input or None if it is variable
        """
        pass


class VectorDataUtil(DataUtil):
    def __init__(self, inputs: pd.DataFrame, outputs: pd.DataFrame, cuda: bool, normalisationMode=normalisation.NormalisationMode.MAX_BY_COLUMN,
            differingOutputNormalisationMode=None):
        """
        :param inputs: the inputs
        :param outputs: the outputs
        :param cuda: whether to apply CUDA
        :param normalisationMode: the normalisation mode to use for inputs and (unless differingOutputNormalisationMode is specified) outputs
        :param differingOutputNormalisationMode: the normalisation mode to apply to outputs
        """
        if inputs.shape[0] != outputs.shape[0]:
            raise ValueError("Output length must be equal to input length")
        self.inputs = inputs
        self.outputs = outputs
        self.normalisationMode = normalisationMode
        self.inputValues = inputs.values
        self.outputValues = outputs.values
        inputScaler = normalisation.VectorDataScaler(self.inputs, self.normalisationMode)
        self.inputValues = inputScaler.getNormalisedArray(self.inputs)
        self.inputTensorScaler = TensorScalerFromVectorDataScaler(inputScaler, cuda)
        outputScaler = normalisation.VectorDataScaler(self.outputs, self.normalisationMode if differingOutputNormalisationMode is None else differingOutputNormalisationMode)
        self.outputValues = outputScaler.getNormalisedArray(self.outputs)
        self.outputTensorScaler = TensorScalerFromVectorDataScaler(outputScaler, cuda)

    def getOutputTensorScaler(self):
        return self.outputTensorScaler

    def getInputTensorScaler(self):
        return self.inputTensorScaler

    def splitInputOutputPairs(self, fractionalSizeOfFirstSet):
        n = self.inputs.shape[0]
        sizeA = int(n * fractionalSizeOfFirstSet)
        indices = list(range(n))
        indices_A = indices[:sizeA]
        indices_B = indices[sizeA:]
        A = self._inputOutputPairs(indices_A)
        B = self._inputOutputPairs(indices_B)
        return A, B

    def _inputOutputPairs(self, indices):
        n = len(indices)
        X = torch.zeros((n, self.inputDim()))
        Y = torch.zeros((n, self.outputDim()), dtype=self._torchOutputDtype())

        for i, outputIdx in enumerate(indices):
            inputData, outputData = self._inputOutputPair(outputIdx)
            if i == 0:
                if inputData.size() != X[i].size():
                    raise Exception(f"Unexpected input size: expected {X[i].size()}, got {inputData.size()}")
                if outputData.size() != Y[i].size():
                    raise Exception(f"Unexpected output size: expected {Y[i].size()}, got {outputData.size()}")
            X[i] = inputData
            Y[i] = outputData

        return X, Y

    def _inputOutputPair(self, idx):
        outputData = torch.from_numpy(self.outputValues[idx, :])
        inputData = torch.from_numpy(self.inputValues[idx, :])
        return inputData, outputData

    def inputDim(self):
        return self.inputs.shape[1]

    def outputDim(self):
        """
        :return: the dimensionality of the outputs (ground truth values)
        """
        return self.outputs.shape[1]

    def modelOutputDim(self):
        return self.outputDim()

    def _torchOutputDtype(self):
        return None  # use default (some float)


class ClassificationVectorDataUtil(VectorDataUtil):
    def __init__(self, inputs: pd.DataFrame, outputs: pd.DataFrame, cuda, numClasses, normalisationMode=normalisation.NormalisationMode.MAX_BY_COLUMN):
        if len(outputs.columns) != 1:
            raise Exception(f"Exactly one output dimension (the class index) is required, got {len(outputs.columns)}")
        super().__init__(inputs, outputs, cuda, normalisationMode=normalisationMode, differingOutputNormalisationMode=normalisation.NormalisationMode.NONE)
        self.numClasses = numClasses

    def modelOutputDim(self):
        return self.numClasses

    def _torchOutputDtype(self):
        return torch.long

    def _inputOutputPairs(self, indices):
        # classifications requires that the second (1-element) dimension be dropped
        inputs, outputs = super()._inputOutputPairs(indices)
        return inputs, outputs.view(outputs.shape[0])


class TorchDataSet:
    @abstractmethod
    def iterBatches(self, batchSize: int, shuffle: bool = False, inputOnly=False) -> Generator[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], None, None]:
        pass

    @abstractmethod
    def size(self) -> Optional[int]:
        pass


class TorchDataSetProvider:
    def __init__(self, inputTensorScaler: Optional[TensorScaler] = None, outputTensorScaler: Optional[TensorScaler] = None,
            inputDim: Optional[int] = None, modelOutputDim: int = None):
        if inputTensorScaler is None:
            inputTensorScaler = TensorScalerIdentity()
        if outputTensorScaler is None:
            outputTensorScaler = TensorScalerIdentity()
        if modelOutputDim is None:
            raise ValueError("The model output dimension must be provided")
        self.inputTensorScaler = inputTensorScaler
        self.outputTensorScaler = outputTensorScaler
        self.inputDim = inputDim
        self.modelOutputDim = modelOutputDim

    @abstractmethod
    def provideSplit(self, fractionalSizeOfFirstSet: float) -> Tuple[TorchDataSet, TorchDataSet]:
        pass

    def getOutputTensorScaler(self) -> TensorScaler:
        return self.outputTensorScaler

    def getInputTensorScaler(self) -> TensorScaler:
        return self.inputTensorScaler

    def getModelOutputDim(self) -> Optional[int]:
        """
        :return: the number of output dimensions that would be required to be generated by the model to match this dataset.
        """
        return self.modelOutputDim

    def getInputDim(self) -> Optional[int]:
        """
        :return: the number of output dimensions that would be required to be generated by the model to match this dataset.
            For models that accept variable input sizes (such as RNNs), this may be None.
        """
        return self.inputDim


class TorchDataSetFromTensors(TorchDataSet):
    def __init__(self, x: torch.Tensor, y: Optional[torch.Tensor], cuda: bool):
        if y is not None and x.shape[0] != y.shape[0]:
            raise ValueError("Tensors are not of the same length")
        self.x = x
        self.y = y
        self.cuda = cuda

    def iterBatches(self, batchSize: int, shuffle: bool = False, inputOnly=False) -> Generator[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], None, None]:
        tensors = (self.x, self.y) if not inputOnly and self.y is not None else (self.x,)
        yield from self._get_batches(tensors, batchSize, shuffle)

    def _get_batches(self, tensors: Sequence[torch.Tensor], batch_size, shuffle):
        length = len(tensors[0])
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            batch = []
            for tensor in tensors:
                if len(tensor) != length:
                    raise Exception("Passed tensors of differing lengths")
                t = tensor[excerpt]
                if self.cuda:
                    t = t.cuda()
                batch.append(Variable(t))
            if len(batch) == 1:
                yield batch[0]
            else:
                yield tuple(batch)
            start_idx += batch_size

    def size(self):
        return self.y.shape[0]


class TorchDataSetProviderFromDataUtil(TorchDataSetProvider):
    def __init__(self, dataUtil: DataUtil, cuda: bool):
        super().__init__(inputTensorScaler=dataUtil.getInputTensorScaler(), outputTensorScaler=dataUtil.getOutputTensorScaler(),
            inputDim=dataUtil.inputDim(), modelOutputDim=dataUtil.modelOutputDim())
        self.dataUtil = dataUtil
        self.cuda = cuda

    def provideSplit(self, fractionalSizeOfFirstSet: float) -> Tuple[TorchDataSet, TorchDataSet]:
        (x1, y1), (x2, y2) = self.dataUtil.splitInputOutputPairs(fractionalSizeOfFirstSet)
        return TorchDataSetFromTensors(x1, y1, self.cuda), TorchDataSetFromTensors(x2, y2, self.cuda)
