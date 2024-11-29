import logging
import math
import random
from abc import ABC, abstractmethod
from typing import Tuple, Sequence, TypeVar, Generic, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import StratifiedShuffleSplit

from ..util.pickle import setstate
from ..util.string import ToStringMixin

log = logging.getLogger(__name__)

T = TypeVar("T")


class BaseInputOutputData(Generic[T], ABC):
    def __init__(self, inputs: T, outputs: T):
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
    def filter_indices(self, indices: Sequence[int]) -> __qualname__:
        pass


class InputOutputArrays(BaseInputOutputData[np.ndarray]):
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        super().__init__(inputs, outputs)

    def filter_indices(self, indices: Sequence[int]) -> __qualname__:
        inputs = self.inputs[indices]
        outputs = self.outputs[indices]
        return InputOutputArrays(inputs, outputs)

    def to_torch_data_loader(self, batch_size=64, shuffle=True):
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError(f"Could not import torch, did you install it?")
        dataset = TensorDataset(torch.tensor(self.inputs), torch.tensor(self.outputs))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class InputOutputData(BaseInputOutputData[pd.DataFrame], ToStringMixin):
    """
    Holds input and output data for learning problems
    """
    def __init__(self, inputs: pd.DataFrame, outputs: pd.DataFrame, weights: Optional[Union[pd.Series, "DataPointWeighting"]] = None):
        super().__init__(inputs, outputs)
        if isinstance(weights, DataPointWeighting):
            weights = weights.compute_weights(inputs, outputs)
        self.weights = weights

    def __setstate__(self, state):
        setstate(InputOutputData, self, state, new_optional_properties=["weights"])

    def _tostring_object_info(self) -> str:
        return f"N={len(self.inputs)}, numInputColumns={len(self.inputs.columns)}, numOutputColumns={len(self.outputs.columns)}"

    @classmethod
    def from_data_frame(cls, df: pd.DataFrame, *output_columns: str) -> "InputOutputData":
        """
        :param df: a data frame containing both input and output columns
        :param output_columns: the output column name(s)
        :return: an InputOutputData instance with inputs and outputs separated
        """
        inputs = df[[c for c in df.columns if c not in output_columns]]
        outputs = df[list(output_columns)]
        return cls(inputs, outputs)

    def to_data_frame(self, add_weights: bool = False, weights_col_name: str = "weights") -> pd.DataFrame:
        """
        :param add_weights: whether to add the weights as a column (provided that weights are present)
        :param weights_col_name: the column name to use for weights if `add_weights` is True
        :return: a data frame containing both the inputs and outputs (and optionally the weights)
        """
        df = pd.concat([self.inputs, self.outputs], axis=1)
        if add_weights and self.weights is not None:
            df[weights_col_name] = self.weights
        return df

    def to_df(self, add_weights: bool = False, weights_col_name: str = "weights") -> pd.DataFrame:
        return self.to_data_frame(add_weights=add_weights, weights_col_name=weights_col_name)

    def filter_indices(self, indices: Sequence[int]) -> __qualname__:
        inputs = self.inputs.iloc[indices]
        outputs = self.outputs.iloc[indices]
        weights = None
        if self.weights is not None:
            weights = self.weights.iloc[indices]
        return InputOutputData(inputs, outputs, weights)

    def filter_index(self, index_elements: Sequence[any]) -> __qualname__:
        inputs = self.inputs.loc[index_elements]
        outputs = self.outputs.loc[index_elements]
        weights = None
        if self.weights is not None:
            weights = self.weights
        return InputOutputData(inputs, outputs, weights)

    @property
    def input_dim(self):
        return self.inputs.shape[1]

    @property
    def output_dim(self):
        return self.outputs.shape[1]

    def compute_input_output_correlation(self):
        correlations = {}
        for outputCol in self.outputs.columns:
            correlations[outputCol] = {}
            output_series = self.outputs[outputCol]
            for inputCol in self.inputs.columns:
                input_series = self.inputs[inputCol]
                pcc, pvalue = scipy.stats.pearsonr(input_series, output_series)
                correlations[outputCol][inputCol] = pcc
        return correlations

    def apply_weighting(self, weighting: "DataPointWeighting"):
        self.weights = weighting.compute_weights(self.inputs, self.outputs)


TInputOutputData = TypeVar("TInputOutputData", bound=BaseInputOutputData)


class DataSplitter(ABC, Generic[TInputOutputData]):
    @abstractmethod
    def split(self, data: TInputOutputData) -> Tuple[TInputOutputData, TInputOutputData]:
        pass


class DataSplitterFractional(DataSplitter):
    def __init__(self, fractional_size_of_first_set: float, shuffle=True, random_seed=42):
        if not 0 <= fractional_size_of_first_set <= 1:
            raise Exception(f"invalid fraction: {fractional_size_of_first_set}")
        self.fractionalSizeOfFirstSet = fractional_size_of_first_set
        self.shuffle = shuffle
        self.randomSeed = random_seed

    def split_with_indices(self, data) -> Tuple[Tuple[Sequence[int], Sequence[int]], Tuple[TInputOutputData, TInputOutputData]]:
        num_data_points = len(data)
        split_index = int(num_data_points * self.fractionalSizeOfFirstSet)
        if self.shuffle:
            rand = np.random.RandomState(self.randomSeed)
            indices = rand.permutation(num_data_points)
        else:
            indices = range(num_data_points)
        indices_a = indices[:split_index]
        indices_b = indices[split_index:]
        a = data.filter_indices(list(indices_a))
        b = data.filter_indices(list(indices_b))
        return (indices_a, indices_b), (a, b)

    def split(self, data: TInputOutputData) -> Tuple[TInputOutputData, TInputOutputData]:
        _, (a, b) = self.split_with_indices(data)
        return a, b


class DataSplitterFromDataFrameSplitter(DataSplitter[InputOutputData]):
    """
    Creates a DataSplitter from a DataFrameSplitter, which can be applied either to the input or the output data.
    It supports only InputOutputData, not other subclasses of BaseInputOutputData.
    """
    def __init__(self, data_frame_splitter: "DataFrameSplitter", fractional_size_of_first_set: float, apply_to_input=True):
        """
        :param data_frame_splitter: the splitter to apply
        :param fractional_size_of_first_set: the desired fractional size of the first set when applying the splitter
        :param apply_to_input: if True, apply the splitter to the input data frame; if False, apply it to the output data frame
        """
        self.dataFrameSplitter = data_frame_splitter
        self.fractionalSizeOfFirstSet = fractional_size_of_first_set
        self.applyToInput = apply_to_input

    def split(self, data: InputOutputData) -> Tuple[InputOutputData, InputOutputData]:
        if not isinstance(data, InputOutputData):
            raise ValueError(f"{self} is only applicable to instances of {InputOutputData.__name__}, got {data}")
        df = data.inputs if self.applyToInput else data.outputs
        indices_a, indices_b = self.dataFrameSplitter.compute_split_indices(df, self.fractionalSizeOfFirstSet)
        a = data.filter_indices(list(indices_a))
        b = data.filter_indices(list(indices_b))
        return a, b


class DataSplitterFromSkLearnSplitter(DataSplitter):
    def __init__(self, sklearn_splitter):
        """
        :param sklearn_splitter: an instance of one of the splitter classes from sklearn.model_selection,
            see https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
        """
        self.sklearn_splitter = sklearn_splitter

    def split(self, data: TInputOutputData) -> Tuple[TInputOutputData, TInputOutputData]:
        splitter_result = self.sklearn_splitter.split(data.inputs, data.outputs)
        split = next(iter(splitter_result))
        first_indices, second_indices = split
        return data.filter_indices(first_indices), data.filter_indices(second_indices)


class DataSplitterStratifiedShuffleSplit(DataSplitterFromSkLearnSplitter):
    def __init__(self, fractional_size_of_first_set: float, random_seed=42):
        super().__init__(StratifiedShuffleSplit(n_splits=1, train_size=fractional_size_of_first_set, random_state=random_seed))

    @staticmethod
    def is_applicable(io_data: InputOutputData):
        class_counts = io_data.outputs.value_counts()
        return all(class_counts >= 2)


class DataFrameSplitter(ABC):
    @abstractmethod
    def compute_split_indices(self, df: pd.DataFrame, fractional_size_of_first_set: float) -> Tuple[Sequence[int], Sequence[int]]:
        pass

    @staticmethod
    def split_with_indices(df: pd.DataFrame, indices_pair: Tuple[Sequence[int], Sequence[int]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        indices_a, indices_b = indices_pair
        a = df.iloc[indices_a]
        b = df.iloc[indices_b]
        return a, b

    def split(self, df: pd.DataFrame, fractional_size_of_first_set: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.split_with_indices(df, self.compute_split_indices(df, fractional_size_of_first_set))


class DataFrameSplitterFractional(DataFrameSplitter):
    def __init__(self, shuffle=False, random_seed=42):
        self.randomSeed = random_seed
        self.shuffle = shuffle

    def compute_split_indices(self, df: pd.DataFrame, fractional_size_of_first_set: float) -> Tuple[Sequence[int], Sequence[int]]:
        n = df.shape[0]
        size_a = int(n * fractional_size_of_first_set)
        if self.shuffle:
            rand = np.random.RandomState(self.randomSeed)
            indices = rand.permutation(n)
        else:
            indices = list(range(n))
        indices_a = indices[:size_a]
        indices_b = indices[size_a:]
        return indices_a, indices_b


class DataFrameSplitterColumnEquivalenceClass(DataFrameSplitter):
    """
    Performs a split that keeps together data points/rows that have the same value in a given column, i.e.
    with respect to that column, the items having the same values are viewed as a unit; they form an equivalence class, and all
    data points belonging to the same class are either in the first set or the second set.

    The split is performed at the level of unique items in the column, i.e. the given fraction of equivalence
    classes will end up in the first set and the rest in the second set.

    The list if unique items in the column can be shuffled before applying the split. If no shuffling is applied,
    the original order in the data frame is maintained, and if the items were grouped by equivalence class in the
    original data frame, the split will correspond to a fractional split without shuffling where the split boundary
    is adjusted to not separate an equivalence class.
    """
    def __init__(self, column: str, shuffle=True, random_seed=42):
        """
        :param column: the column which defines the equivalence classes (groups of data points/rows that must not be separated)
        :param shuffle: whether to shuffle the list of unique values in the given column before applying the split
        :param random_seed:
        """
        self.column = column
        self.shuffle = shuffle
        self.random_seed = random_seed

    def compute_split_indices(self, df: pd.DataFrame, fractional_size_of_first_set: float) -> Tuple[Sequence[int], Sequence[int]]:
        values = list(df[self.column].unique())
        if self.shuffle:
            rng = random.Random(self.random_seed)
            rng.shuffle(values)

        num_items_in_first_set = round(fractional_size_of_first_set * len(values))
        first_set_values = set(values[:num_items_in_first_set])

        first_set_indices = []
        second_set_indices = []
        for i, t in enumerate(df.itertuples()):
            if getattr(t, self.column) in first_set_values:
                first_set_indices.append(i)
            else:
                second_set_indices.append(i)
        return first_set_indices, second_set_indices


class DataPointWeighting(ABC):
    @abstractmethod
    def compute_weights(self, x: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
        pass


class DataPointWeightingRegressionTargetIntervalTotalWeight(DataPointWeighting):
    """
    Based on relative weights specified for intervals of the regression target,
    will weight individual data point weights such that the sum of weights of data points within each interval
    satisfies the user-specified relative weight, while ensuring that the total weight of all data points
    is still equal to the number of data points.

    For example, if one specifies `interval_weights` as [(0.5, 1), (inf, 2)], then the data points with target values
    up to 0.5 will get 1/3 of the weight and the remaining data points will get 2/3 of the weight.
    So if there are 100 data points and 50 of them are in the first interval (up to 0.5), then these 50 data points
    will each get weight 1/3*100/50=2/3 and the remaining 50 data points will each get weight 2/3*100/50=4/3.
    The sum of all weights is the number of data points, i.e. 100.

    Example:

    >>> targets = [0.1, 0.2, 0.5, 0.7, 0.8, 0.6]
    >>> x = pd.DataFrame({"foo": np.zeros(len(targets))})
    >>> y = pd.DataFrame({"target": targets})
    >>> weighting = DataPointWeightingRegressionTargetIntervalTotalWeight([(0.5, 1), (1.0, 2)])
    >>> weights = weighting.compute_weights(x, y)
    >>> assert(np.isclose(weights.sum(), len(y)))
    >>> weights.tolist()
    [0.6666666666666666,
     0.6666666666666666,
     0.6666666666666666,
     1.3333333333333333,
     1.3333333333333333,
     1.3333333333333333]
    """
    def __init__(self, intervals_weights: Sequence[Tuple[float, float]]):
        """
        :param intervals_weights: a sequence of tuples (upper_bound, rel_total_weight) where upper_bound is the upper bound
            of the interval, `(lower_bound, upper_bound]`; `lower_bound` is the upper bound of the preceding interval
            or -inf for the first interval. `rel_total_weight` specifies the relative weight of all data points within
            the interval.
        """
        a = -math.inf
        sum_rel_weights = sum(t[1] for t in intervals_weights)
        self.intervals = []
        for b, rel_weight in intervals_weights:
            self.intervals.append(self.Interval(a, b, rel_weight / sum_rel_weights))
            a = b

    class Interval:
        def __init__(self, a: float, b: float, weight_fraction: float):
            self.a = a
            self.b = b
            self.weight_fraction = weight_fraction

        def contains(self, x: float):
            return self.a < x <= self.b

    def compute_weights(self, x: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
        assert len(y.columns) == 1, f"Only a single regression target is supported {self.__class__.__name__}"
        targets = y.iloc[:, 0]
        n = len(x)
        weights = np.zeros(n)
        num_weighted = 0
        for interval in self.intervals:
            mask = np.array([interval.contains(x) for x in targets])
            subset_size = mask.sum()
            num_weighted += subset_size
            weights[mask] = interval.weight_fraction * n / subset_size
        if num_weighted != n:
            raise Exception("Not all data points were weighted. Most likely, the intervals do not cover the entire range of targets")
        return pd.Series(weights, index=x.index)
