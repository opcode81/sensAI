"""
This module contains sample datasets to facilitate testing and development.
"""
from abc import ABC, abstractmethod

import sklearn.datasets

from sensai.data import InputOutputData
import pandas as pd


class DataSet(ABC):
    @abstractmethod
    def load_io_data(self) -> InputOutputData:
        pass


class DataSetClassificationIris(DataSet):
    def load_io_data(self) -> InputOutputData:
        iris_data = sklearn.datasets.load_iris()
        input_df = pd.DataFrame(iris_data["data"], columns=iris_data["feature_names"]).reset_index(drop=True)
        output_df = pd.DataFrame({"class": [iris_data["target_names"][idx] for idx in iris_data["target"]]}) \
            .reset_index(drop=True)
        return InputOutputData(input_df, output_df)
