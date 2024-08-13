"""
This module contains sample datasets to facilitate testing and development.
"""
from abc import ABC, abstractmethod

import sklearn.datasets

from sensai.data import InputOutputData
import pandas as pd

from sensai.util.string import ToStringMixin


class DataSet(ToStringMixin, ABC):
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


class DataSetClassificationTitanicSurvival(DataSet):
    URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

    COL_INDEX = "PassengerId"
    """
    unique identifier for each passenger
    """
    COL_SURVIVAL = "Survived"
    """
    0 = No, 1 = Yes
    """
    COL_NAME = "Name"
    """
    passenger name
    """
    COL_PASSENGER_CLASS = "Pclass"
    """
    Ticket class as an integer (1 = first, 2 = second, 3 = third)
    """
    COL_SEX = "Sex"
    """
    'male' or 'female'
    """
    COL_AGE_YEARS = "Age"
    """
    age in years (integer)
    """
    COL_SIBLINGS_SPOUSES = "SibSp"
    """
    number of siblings/spouses aboard the Titanic
    """
    COL_PARENTS_CHILDREN = "Parch"
    """
    number of parents/children aboard the Titanic
    """
    COL_FARE_PRICE = "Fare"
    """
    amount of money paid for the ticket
    """
    COL_CABIN = "Cabin"
    """
    the cabin number (if available)
    """
    COL_PORT_EMBARKED = "Embarked"
    """
    port of embarkation ('C' = Cherbourg, 'Q' = Queenstown, 'S' = Southampton)
    """
    COL_TICKET = "Ticket"
    """
    the ticket number
    """
    COLS_METADATA = [COL_NAME, COL_TICKET, COL_CABIN]
    """
    list of columns containing meta-data which are not useful for generalising prediction models
    """

    def __init__(self, drop_metadata_columns: bool = False):
        """
        :param drop_metadata_columns: whether to drop meta-data columns which are not useful for a
            generalising prediction model
        """
        self.drop_metadata_columns = drop_metadata_columns

    def load_io_data(self) -> InputOutputData:
        df = pd.read_csv(self.URL).set_index(self.COL_INDEX, drop=True)
        if self.drop_metadata_columns:
            df.drop(columns=self.COLS_METADATA, inplace=True)
        return InputOutputData.from_data_frame(df, self.COL_SURVIVAL)
