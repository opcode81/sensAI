from abc import ABC, abstractmethod

import pandas as pd
import torch

from .. import normalisation


class TensorScaler(ABC):
    @abstractmethod
    def cuda(self):
        """
        Makes this scaler's components use CUDA
        """
        pass

    def normalise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies scaling/normalisation to the given tensor
        :param tensor: the tensor to scale/normalise
        :return: the scaled/normalised tensor
        """
        pass

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


class DataUtil(ABC):
    """Interface for DataUtil classes, which are used to process data for neural networks"""

    @abstractmethod
    def splitInputOutputPairs(self, fractionalSizeOfFirstSet):
        """
        Splits the data set

        :param fractionalSizeOfFirstSet: the desired fractional size in

        :return: a tuple (A, B, meta) where A and B are tuples (in, out) with input and output data,
            and meta is a dictionary containing meta-data on the split, which may contain the following keys:

            * "infoText": text on the data set/the split performed;
            * "outputIndicesA": output index sequence in the set A (one index for every input/output element of A);
            * "outputIndicesB": output index sequence in the set A;

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
        meta = dict(outputIndicesA=indices_A, outputIndicesB=indices_B)
        return A, B, meta

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
        """
        :return: the dimensionality that is to be output by the model to be trained
        """
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
