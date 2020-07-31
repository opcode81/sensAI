from abc import ABC, abstractmethod
from typing import Tuple, Sequence

import numpy as np
import pandas as pd
import scipy.stats


class InputOutputData:
    def __init__(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        if len(inputs) != len(outputs):
            raise ValueError("Lengths do not match")
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    @property
    def inputDim(self):
        return self.inputs.shape[1]

    @property
    def outputDim(self):
        return self.outputs.shape[1]

    def filterIndices(self, indices: Sequence[int]) -> 'InputOutputData':
        inputs = self.inputs.iloc[indices]
        outputs = self.outputs.iloc[indices]
        return InputOutputData(inputs, outputs)

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


class DataSplitter(ABC):
    @abstractmethod
    def split(self, data: InputOutputData) -> Tuple[InputOutputData, InputOutputData]:
        pass


class DataSplitterFractional(DataSplitter):
    def __init__(self, fractionalSizeOfFirstSet, shuffle=True, randomSeed: int = 42):
        if not 0 <= fractionalSizeOfFirstSet <= 1:
            raise Exception(f"invalid fraction: {fractionalSizeOfFirstSet}")
        self.fractionalSizeOfFirstSet = fractionalSizeOfFirstSet
        self.shuffle = shuffle
        self.randomSeed = randomSeed

    def split(self, data: InputOutputData) -> Tuple[InputOutputData, InputOutputData]:
        numDataPoints = len(data)
        splitIndex = int(numDataPoints * self.fractionalSizeOfFirstSet)
        rand = np.random.RandomState(self.randomSeed)
        if self.shuffle:
            indices = rand.permutation(numDataPoints)
        else:
            indices = range(numDataPoints)
        indicesA = indices[:splitIndex]
        indicesB = indices[splitIndex:]
        A = data.filterIndices(list(indicesA))
        B = data.filterIndices(list(indicesB))
        return A, B
