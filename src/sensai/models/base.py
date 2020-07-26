from abc import ABC, abstractmethod
import pandas as pd


class NamedModel(ABC):
    @abstractmethod
    def getName(self) -> str:
        pass


class PredictorModel(NamedModel, ABC):
    """
    Base class for models that map vectors to predictions
    """
    @abstractmethod
    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def getPredictedVariableNames(self):
        pass

    @abstractmethod
    def isRegressionModel(self) -> bool:
        pass


class FitterModel(NamedModel, ABC):
    """
    Base class for models that can be fitted
    """
    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def isFitted(self) -> bool:
        pass


class FitPredictModel(FitterModel, PredictorModel, ABC):
    pass
