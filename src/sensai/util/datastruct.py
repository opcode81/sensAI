from enum import Enum
from typing import Sequence, Optional, TypeVar, Generic, Tuple, Dict, Any

import pandas as pd

from . import sequences as array_util
from .string import ToStringMixin, dictString

TKey = TypeVar("TKey")
TValue = TypeVar("TValue")


class Trivalent(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    @classmethod
    def fromBool(cls, b: bool):
        return cls.TRUE if b else cls.FALSE

    def isTrue(self):
        return self == Trivalent.TRUE

    def isFalse(self):
        return self == Trivalent.FALSE


class DeferredParams(ToStringMixin):
    """
    Represents a dictionary of parameters that is specifically designed to hold parameters that can only defined late within
    a process (i.e. not initially at construction time), e.g. because the parameters are data-dependent and therefore can only
    be determined once the data has been seen.
    """
    UNDEFINED = "__undefined__DeferredParams"

    def __init__(self):
        self.params = {}

    def _toStringObjectInfo(self) -> str:
        return dictString(self.params)

    def setParam(self, name: str, value: Any):
        self.params[name] = value

    def getParam(self, name, default=UNDEFINED):
        """
        :param name: the parameter name
        :param default: in case no value is set, return this value, and if UNDEFINED (default), raise KeyError
        :return: the parameter value
        """
        if default == self.UNDEFINED:
            return self.params[name]
        else:
            return self.params.get(name, default)

    def getDict(self) -> Dict[str, Any]:
        return self.params


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

    def closestIndex(self, value) -> Optional[int]:
        """
        Finds the index of the value that is closest to the given value.
        If two subsequent values have the same distance, the smaller index is returned.

        :param value: the value to search for
        :return: the index or None if this object is empty
        """
        return array_util.closestIndex(self.values, value)

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

    def closestValue(self, value) -> Optional[TValue]:
        """
        Finds the value that is closest to the given value.
        If two subsequent values have the same distance, the smaller value is returned.

        :param value: the value to search for
        :return: the value or None if this object is empty
        """
        return self._value(self.closestIndex(value))

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

    @classmethod
    def fromSeries(cls, s: pd.Series):
        """
        Creates an instance from a pandas Series, using the series' index as the keys and its values as the values

        :param s: the series
        :return: an instance
        """
        return cls(s.index, s.values)

    def floorIndex(self, key) -> Optional[int]:
        """
        Finds the rightmost index where the key value is less than or equal to the given value

        :param key: the value to search for
        :return: the index or None if there is no such index
        """
        return array_util.floorIndex(self.keys, key)

    def ceilIndex(self, key) -> Optional[int]:
        """
        Finds the leftmost index where the key value is greater than or equal to the given value

        :param key: the value to search for
        :return: the index or None if there is no such index
        """
        return array_util.ceilIndex(self.keys, key)

    def closestIndex(self, key) -> Optional[int]:
        """
        Finds the index where the key is closest to the given value.
        If two subsequent keys have the same distance, the smaller index is returned.

        :param key: the value to search for
        :return: the index or None if this object is empty.
        """
        return array_util.closestIndex(self.keys, key)

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

    def closestValue(self, key) -> Optional[TValue]:
        """
        Finds the value that is closest to the given value.
        If two subsequent values have the same distance, the smaller value is returned.

        :param value: the value to search for
        :return: the value or None if this object is empty
        """
        return array_util.closestValue(self.keys, key, values=self.values)

    def floorKeyAndValue(self, key) -> Optional[Tuple[TKey, TValue]]:
        idx = self.floorIndex(key)
        return None if idx is None else (self.keys[idx], self.values[idx])

    def ceilKeyAndValue(self, key) -> Optional[Tuple[TKey, TValue]]:
        idx = self.ceilIndex(key)
        return None if idx is None else (self.keys[idx], self.values[idx])

    def closestKeyAndValue(self, key) -> Optional[Tuple[TKey, TValue]]:
        idx = self.closestIndex(key)
        return None if idx is None else (self.keys[idx], self.values[idx])

    def valueSliceInner(self, lowerBoundKey, upperBoundKey):
        return array_util.valueSliceInner(self.keys, lowerBoundKey, upperBoundKey, values=self.values)

    def valueSliceOuter(self, lowerBoundKey, upperBoundKey, fallback=False):
        return array_util.valueSliceOuter(self.keys, lowerBoundKey, upperBoundKey, values=self.values, fallbackBounds=fallback)


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

    def floorKeyAndValue(self, key) -> Optional[Tuple[TKey, TValue]]:
        idx = self.floorIndex(key)
        return None if idx is None else self.entries[idx]

    def ceilIndex(self, key) -> Optional[int]:
        """Find leftmost index where the key is greater than or equal to the given key"""
        return self._sortedKeys.ceilIndex(key)

    def ceilValue(self, key) -> Optional[TValue]:
        return self._value(self.ceilIndex(key))

    def ceilKeyAndValue(self, key) -> Optional[Tuple[TKey, TValue]]:
        idx = self.ceilIndex(key)
        return None if idx is None else self.entries[idx]

    def closestIndex(self, key) -> Optional[int]:
        return self._sortedKeys.closestIndex(key)

    def closestValue(self, key) -> Optional[TValue]:
        return self._value(self.closestIndex(key))

    def closestKeyAndValue(self, key):
        idx = self.closestIndex(key)
        return None if idx is None else self.entries[idx]

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