from abc import ABC, abstractmethod
from typing import Tuple, Sequence, TypeVar, List, Generic

import numpy as np
import pandas as pd
import scipy.stats


class BaseInputOutputData(ABC):
    def __init__(self, inputs, outputs):
        """
        :param inputs: expected to have shape and __len__
        :param outputs: expected to have shape and __len__
        """
        if len(inputs) != len(outputs):
            raise ValueError("Lengths do not match")
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    @abstractmethod
    def filterIndices(self, indices: Sequence[int]) -> __qualname__:
        pass


class InputOutputArrays(BaseInputOutputData):
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        super().__init__(inputs, outputs)

    def filterIndices(self, indices: Sequence[int]) -> __qualname__:
        inputs = self.inputs[indices]
        outputs = self.outputs[indices]
        return InputOutputArrays(inputs, outputs)

    def toTorchDataLoader(self, batchSize=64, shuffle=True):
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError(f"Could not import torch, did you install it?")
        dataSet = TensorDataset(torch.tensor(self.inputs), torch.tensor(self.outputs))
        return DataLoader(dataSet, batch_size=batchSize, shuffle=shuffle)


# TODO: Rename to InputOutputDataFrames when the time for breaking changes has come
class InputOutputData(BaseInputOutputData):
    def __init__(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        super().__init__(inputs, outputs)

    def filterIndices(self, indices: Sequence[int]) -> __qualname__:
        inputs = self.inputs.iloc[indices]
        outputs = self.outputs.iloc[indices]
        return InputOutputData(inputs, outputs)

    @property
    def inputDim(self):
        return self.inputs.shape[1]

    @property
    def outputDim(self):
        return self.outputs.shape[1]

    def computeInputOutputCorrelation(self):
        correlations = {}
        for outputCol in self.outputs.columns:
            correlations[outputCol] = {}
            outputSeries = self.outputs[outputCol]
            for inputCol in self.inputs.columns:
                inputSeries = self.inputs[inputCol]
                pcc, pvalue = scipy.stats.pearsonr(inputSeries, outputSeries)
                correlations[outputCol][inputCol] = pcc
        return correlations


TInputOutputData = TypeVar("TInputOutputData", bound=BaseInputOutputData)


class DataSplitter(ABC, Generic[TInputOutputData]):
    @abstractmethod
    def split(self, data: TInputOutputData) -> Tuple[TInputOutputData, TInputOutputData]:
        pass


class DataSplitterFractional(DataSplitter):
    def __init__(self, fractionalSizeOfFirstSet: float, shuffle=True, randomSeed=42):
        if not 0 <= fractionalSizeOfFirstSet <= 1:
            raise Exception(f"invalid fraction: {fractionalSizeOfFirstSet}")
        self.fractionalSizeOfFirstSet = fractionalSizeOfFirstSet
        self.shuffle = shuffle
        self.randomSeed = randomSeed

    def split(self, data: TInputOutputData) -> Tuple[TInputOutputData, TInputOutputData]:
        numDataPoints = len(data)
        splitIndex = int(numDataPoints * self.fractionalSizeOfFirstSet)
        if self.shuffle:
            rand = np.random.RandomState(self.randomSeed)
            indices = rand.permutation(numDataPoints)
        else:
            indices = range(numDataPoints)
        indicesA = indices[:splitIndex]
        indicesB = indices[splitIndex:]
        A = data.filterIndices(list(indicesA))
        B = data.filterIndices(list(indicesB))
        return A, B


class DataSplitterFromDataFrameSplitter(DataSplitter[InputOutputData]):
    """
    Creates a DataSplitter from a DataFrameSplitter, which can be applied either to the input or the output data.
    It supports only InputOutputData, not other subclasses of BaseInputOutputData.
    """
    def __init__(self, dataFrameSplitter: "DataFrameSplitter", fractionalSizeOfFirstSet: float, applyToInput=True):
        """
        :param dataFrameSplitter: the splitter to apply
        :param fractionalSizeOfFirstSet: the desired fractional size of the first set when applying the splitter
        :param applyToInput: if True, apply the splitter to the input data frame; if False, apply it to the output data frame
        """
        self.dataFrameSplitter = dataFrameSplitter
        self.fractionalSizeOfFirstSet = fractionalSizeOfFirstSet
        self.applyToInput = applyToInput

    def split(self, data: InputOutputData) -> Tuple[InputOutputData, InputOutputData]:
        if not isinstance(data, InputOutputData):
            raise ValueError(f"{self} is only applicable to instances of {InputOutputData.__name__}, got {data}")
        df = data.inputs if self.applyToInput else data.outputs
        indicesA, indicesB = self.dataFrameSplitter.computeSplitIndices(df, self.fractionalSizeOfFirstSet)
        A = data.filterIndices(list(indicesA))
        B = data.filterIndices(list(indicesB))
        return A, B


class DataFrameSplitter(ABC):
    @abstractmethod
    def computeSplitIndices(self, df: pd.DataFrame, fractionalSizeOfFirstSet: float) -> Tuple[Sequence[int], Sequence[int]]:
        pass

    @staticmethod
    def split(df: pd.DataFrame, indicesPair: Tuple[Sequence[int], Sequence[int]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        indicesA, indicesB = indicesPair
        A = df.iloc[indicesA]
        B = df.iloc[indicesB]
        return A, B


class DataFrameSplitterFractional(DataFrameSplitter):
    def __init__(self, shuffle=False, randomSeed=42):
        self.randomSeed = randomSeed
        self.shuffle = shuffle

    def computeSplitIndices(self, df: pd.DataFrame, fractionalSizeOfFirstSet: float) -> Tuple[Sequence[int], Sequence[int]]:
        n = df.shape[0]
        sizeA = int(n * fractionalSizeOfFirstSet)
        if self.shuffle:
            rand = np.random.RandomState(self.randomSeed)
            indices = rand.permutation(n)
        else:
            indices = list(range(n))
        indices_A = indices[:sizeA]
        indices_B = indices[sizeA:]
        return indices_A, indices_B
