import collections
from typing import Hashable, Dict

from .string import ToStringMixin


class RelativeFrequencyCounter(ToStringMixin):
    """
    Counts the absolute and relative frequency of an event
    """
    def __init__(self):
        self.numTotal = 0
        self.numRelevant = 0

    def count(self, isRelevantEvent) -> None:
        """
        Adds to the the count.
        The nominator is incremented only if we are counting a relevant event.
        The denominator is always incremented.

        :param isRelevantEvent: whether we are counting a relevant event
        """
        self.numTotal += 1
        if isRelevantEvent:
            self.numRelevant += 1

    def _toStringObjectInfo(self):
        info = f"{self.numRelevant}/{self.numTotal}"
        if self.numTotal > 0:
            info += f", {100 * self.numRelevant / self.numTotal:.2f}%"
        return info

    def add(self, relativeFrequencyCounter: __qualname__) -> None:
        """
        Adds the counts of the given counter to this object

        :param relativeFrequencyCounter: the counter whose data to add
        """
        self.numTotal += relativeFrequencyCounter.numTotal
        self.numRelevant += relativeFrequencyCounter.numRelevant

    def getRelativeFrequency(self) -> float:
        return self.numRelevant / self.numTotal


class DistributionCounter(ToStringMixin):
    """
    Supports the counting of the frequencies with which (mutually exclusive) events occur
    """
    def __init__(self):
        self.counts = collections.defaultdict(lambda: 0)
        self.totalCount = 0

    def count(self, event: Hashable) -> None:
        """
        Increments the count of the given event

        :param event: the event/key whose count to increment, which must be hashable
        """
        self.totalCount += 1
        self.counts[event] += 1

    def getDistribution(self) -> Dict[Hashable, float]:
        """
        :return: a dictionary mapping events (as previously passed to count) to their relative frequencies
        """
        return {k: v/self.totalCount for k, v in self.counts.items()}

    def _toStringObjectInfo(self):
        return ", ".join([f"{str(k)}: {v} ({v/self.totalCount:.3f})" for k, v in self.counts.items()])
