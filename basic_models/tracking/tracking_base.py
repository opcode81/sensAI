from abc import ABC, abstractmethod
from typing import Dict, Any


class TrackedExperiment(ABC):
    def __init__(self, additionalLoggingValuesDict=None):
        """
        Base class for tracking
        :param additionalLoggingValuesDict: additional values to be logged for each run
        """
        self.additionalLoggingValuesDict = additionalLoggingValuesDict

    def trackValues(self, valuesDict: Dict[str, Any]):
        valuesDict = dict(valuesDict)
        if self.additionalLoggingValuesDict is not None:
            valuesDict.update(self.additionalLoggingValuesDict)
        self._trackValues(valuesDict)

    @abstractmethod
    def _trackValues(self, valuesDict):
        pass


class TrackedExperimentDataProvider(ABC):
    @abstractmethod
    def setTrackedExperiment(self, trackedExperiment: TrackedExperiment):
        pass
