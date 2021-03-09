from bisect import bisect_right, bisect_left
from typing import Sequence, Optional, TypeVar, Generic, Tuple

from . import sequences as array_util

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
        return array_util.floorIndex(self.values, value)

    def ceilIndex(self, value) -> Optional[int]:
        """
        Finds the leftmost index where the value is greater than or equal to the given value

        :param value: the value to search for
        :return: the index or None if there is no such index
        """
        return array_util.ceilIndex(self.values, value)

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


class SortedKeysAndValues(Generic[TKey, TValue]):
    def __init__(self, keys: Sequence[TKey], values: Sequence[TValue]):
        """
        :param keys: a sorted sequence of keys
        :param values: a sequence of corresponding values
        """
        if len(keys) != len(values):
            raise ValueError("Lengths of keys and values do not match")
        self.keys = keys
        self.values = values

    def floorIndex(self, value) -> Optional[int]:
        """
        Finds the rightmost index where the value is less than or equal to the given value

        :param value: the value to search for
        :return: the index or None if there is no such index
        """
        return array_util.floorIndex(self.values, value)

    def ceilIndex(self, value) -> Optional[int]:
        """
        Finds the leftmost index where the value is greater than or equal to the given value

        :param value: the value to search for
        :return: the index or None if there is no such index
        """
        return array_util.ceilIndex(self.values, value)

    def floorValue(self, key) -> Optional[TValue]:
        """
        Returns the value for the largest index where the corresponding key is less than or equal to the given value

        :param value: the value to search for
        :return: the value or None if there is no such value
        """
        return array_util.floorValue(self.keys, key, values=self.values)

    def ceilValue(self, key) -> Optional[TValue]:
        """
        Returns the value for the smallest index where the corresponding key is greater than or equal to the given value

        :param value: the value to search for
        :return: the value or None if there is no such value
        """
        return array_util.ceilValue(self.keys, key, values=self.values)

    def valueSliceInner(self, lowerBoundKey, upperBoundKey):
        return array_util.valueSliceOuter(self.keys, lowerBoundKey, upperBoundKey, values=self.values)

    def valueSliceOuter(self, lowerBoundKey, upperBoundKey):
        return array_util.valueSliceOuter(self.keys, lowerBoundKey, upperBoundKey, values=self.values)


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

    def slice(self, lowerBoundKey=None, upperBoundKey=None, inner=True) -> "SortedKeyValuePairs":
        """
        :param lowerBoundKey: the key defining the start of the slice (depending on inner);
            if None, the first included entry will be the very first entry
        :param upperBoundKey: the key defining the end of the slice (depending on inner);
            if None, the last included entry will be the very last entry
        :param inner: if True, the returned slice will be within the bounds; if False, the the returned
            slice is extended by one entry in both directions such that it contains the bounds (where possible)
        :return:
        """
        if lowerBoundKey is not None:
            fromIndex = self.ceilIndex(lowerBoundKey) if inner else self.floorIndex(lowerBoundKey)
            if fromIndex is None:
                fromIndex = 0
        else:
            fromIndex = 0
        if upperBoundKey is not None:
            toIndex = self.floorIndex(upperBoundKey) if inner else self.ceilIndex(upperBoundKey)
            if toIndex is None:
                toIndex = len(self.entries)
        else:
            toIndex = len(self.entries)
        return SortedKeyValuePairs(self.entries[fromIndex:toIndex])