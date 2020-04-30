from bisect import bisect_right, bisect_left
from typing import Sequence, Optional, TypeVar, Generic, Tuple

TKey = TypeVar("TKey")
TValue = TypeVar("TValue")


class SortedValues(Generic[TValue]):
    """
    Provides convenient binary search (bisection) operations for sorted sequences
    """
    def __init__(self, sortedValues: Sequence[TValue]):
        self.values = sortedValues

    def __len__(self):
        return len(self.values)

    def floorIndex(self, value) -> Optional[int]:
        """
        Finds the rightmost index where the value is less than or equal to the given value

        :param value: the value to search for
        :return: the index or None if there is no such index
        """
        idx = bisect_right(self.values, value)
        if idx:
            return idx - 1
        return None

    def ceilIndex(self, value) -> Optional[int]:
        """
        Finds the leftmost index where the value is greater than or equal to the given value

        :param value: the value to search for
        :return: the index or None if there is no such index
        """
        idx = bisect_left(self.values, value)
        if idx != len(self.values):
            return idx
        return None

    def _value(self, idx: Optional[int]) -> Optional[TValue]:
        if idx is None:
            return None
        else:
            return self.values[idx]

    def floorValue(self, value) -> Optional[TValue]:
        """
        Finds the largest value that is less than or equal to the given value

        :param value: the value to search for
        :return: the value or None if there is no such value
        """
        return self._value(self.floorIndex(value))

    def ceilValue(self, value) -> Optional[TValue]:
        """
        Finds the smallest value that is greater than or equal to the given value

        :param value: the value to search for
        :return: the value or None if there is no such value
        """
        return self._value(self.ceilIndex(value))

    def _valueSlice(self, firstIndex, lastIndex):
        if firstIndex is None or lastIndex is None:
            return None
        return self.values[firstIndex:lastIndex+1]

    def valueSlice(self, lowestKey, highestKey) -> Optional[Sequence[TValue]]:
        return self._valueSlice(self.ceilIndex(lowestKey), self.floorIndex(highestKey))


class SortedKeyValuePairs(Generic[TKey, TValue]):
    @classmethod
    def fromUnsortedKeyValuePairs(cls, unsortedKeyValuePairs: Sequence[Tuple[TKey, TValue]]):
        return cls(sorted(unsortedKeyValuePairs, key=lambda x: x[0]))

    def __init__(self, sortedKeyValuePairs: Sequence[Tuple[TKey, TValue]]):
        self.entries = sortedKeyValuePairs
        self._sortedKeys = SortedValues([t[0] for t in sortedKeyValuePairs])

    def _value(self, idx: Optional[int]) -> Optional[TValue]:
        if idx is None:
            return None
        return self.entries[idx][1]

    def floorIndex(self, key) -> Optional[int]:
        """Finds the rightmost index where the key is less than or equal to the given key"""
        return self._sortedKeys.floorIndex(key)

    def floorValue(self, key) -> Optional[TValue]:
        return self._value(self.floorIndex(key))

    def ceilIndex(self, key) -> Optional[int]:
        """Find leftmost index where the key is greater than or equal to the given key"""
        return self._sortedKeys.ceilIndex(key)

    def ceilValue(self, key) -> Optional[TValue]:
        return self._value(self.ceilIndex(key))

    def _valueSlice(self, firstIndex, lastIndex):
        if firstIndex is None or lastIndex is None:
            return None
        return [e[1] for e in self.entries[firstIndex:lastIndex+1]]

    def valueSlice(self, lowestKey, highestKey) -> Optional[Sequence[TValue]]:
        return self._valueSlice(self.ceilIndex(lowestKey), self.floorIndex(highestKey))
