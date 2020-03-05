from bisect import bisect_right, bisect_left
from typing import Sequence, Optional, TypeVar, Generic, Tuple

TKey = TypeVar("TKey")
TValue = TypeVar("TValue")


class SortedKeyValuePairs(Generic[TKey, TValue]):
    @classmethod
    def fromUnsortedKeyValuePairs(cls, unsortedKeyValuePairs: Sequence[Tuple[TKey, TValue]]):
        return cls(sorted(unsortedKeyValuePairs, key=lambda x: x[0]))

    def __init__(self, sortedKeyValuePairs: Sequence[Tuple[TKey, TValue]]):
        self.entries = sortedKeyValuePairs
        self.keys = [t[0] for t in sortedKeyValuePairs]

    def _value(self, idx: Optional[int]) -> Optional[TValue]:
        if idx is None:
            return None
        return self.entries[idx][1]

    def floorIndex(self, key) -> Optional[int]:
        """Finds the rightmost index less than or equal to key"""
        idx = bisect_right(self.keys, key)
        if idx:
            return idx - 1
        return None

    def floorValue(self, key) -> Optional[TValue]:
        return self._value(self.floorIndex(key))

    def ceilIndex(self, key) -> Optional[int]:
        """Find leftmost index where the key is greater than or equal to the given key"""
        idx = bisect_left(self.keys, key)
        if idx != len(self.keys):
            return idx
        return None

    def ceilValue(self, key) -> Optional[TValue]:
        return self._value(self.ceilIndex(key))

    def _valueSlice(self, firstIndex, lastIndex):
        if firstIndex is None or lastIndex is None:
            return None
        return [e[1] for e in self.entries[firstIndex:lastIndex+1]]

    def valueSlice(self, lowestKey, highestKey) -> Optional[Sequence[TValue]]:
        return self._valueSlice(self.ceilIndex(lowestKey), self.floorIndex(highestKey))
