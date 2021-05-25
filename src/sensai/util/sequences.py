from bisect import bisect_right, bisect_left
from typing import Optional, TypeVar, Sequence, Any

T = TypeVar("T")


def getFirstDuplicate(seq: Sequence[T]) -> Optional[T]:
    """
    Returns the first duplicate in a sequence or None

    :param seq: a sequence of hashable elements
    :return:
    """
    setOfElems = set()
    for elem in seq:
        if elem in setOfElems:
            return elem
        setOfElems.add(elem)


def floorIndex(arr, value) -> Optional[int]:
    """
    Finds the rightmost/largest index in a sorted array where the value is less than or equal to the given value

    :param arr: the sorted array of values
    :param value: the value to search for
    :return: the index or None if there is no such index
    """
    idx = bisect_right(arr, value)
    if idx:
        return idx - 1
    return None


def ceilIndex(arr, value) -> Optional[int]:
    """
    finds the leftmost/lowest index in a sorted array where the value is greater than or equal to the given value

    :param arr: the sorted array of values
    :param value: the value to search for
    :return: the index or None if there is no such index
    """
    idx = bisect_left(arr, value)
    if idx != len(arr):
        return idx
    return None


def ceilValue(keys, keyValue, values=None) -> Optional[Any]:
    """
    For a sorted array of keys (and a an array of corresponding values),
    returns the corresponding value (values[i]) for the lowest index i where keys[i] >= keyValue

    :param keys: the sorted array of keys for in which to search for the value closest to `keyValue`
    :param keyValue: the key to search for
    :param values: the array from which to retrieve the value; if None, use `keys`
    :return: the value or None if no such value exists
    """
    idx = ceilIndex(keys, keyValue)
    if idx is None:
        return None
    values = values if values is not None else keys
    return values[idx]


def floorValue(keys, keyValue, values=None, fallbackFirst=False) -> Optional[Any]:
    """
    For a sorted array of keys (and an array of corresponding values),
    returns the corresponding value (values[i]) for the largest index i in keys where keys[i] <= keyValue.

    :param keys: the sorted array of keys in which to perform the lookup
    :param keyValue: the key for which to perform the lookup
    :param values: the sorted array of values; if None, use keys as values
    :param fallbackFirst: if True, then return the first value instead of None for the case where no floor value exists
    :return: the floor value
    """
    idx = floorIndex(keys, keyValue)
    values = values if values is not None else keys
    if idx is None:
        if fallbackFirst:
            return values[0]
        return None
    return values[idx]


def valueSliceInner(keys, lowerBoundKey, upperBoundKey, values=None):
    """
    For a sorted array of keys (and a an array of corresponding values),
    finds indices i, j such that i is the lowest key where keys[i] >= lowerBoundKey and
    j is the largest key where keys[j] <= upperBoundKey,
    and returns the corresponding slice of values values[i:j+1],
    i.e. the slice will not include the bounds keys if they are not present in the keys array.

    :param keys: the sorted array of key values
    :param lowerBoundKey: the key value defining the lower bound
    :param upperBoundKey: the key value defining the upper bound
    :param values: the sorted array of values; if None, use keys
    :return: the corresponding slice of `values`
    """
    if values is None:
        values = keys
    elif len(values) != len(keys):
        raise ValueError("Length of values must match length of keys")
    if lowerBoundKey > upperBoundKey:
        raise ValueError(f"Invalid bounds: {lowerBoundKey} to {upperBoundKey}")
    firstIdx = ceilIndex(keys, lowerBoundKey)
    lastIdx = floorIndex(keys, upperBoundKey)
    if firstIdx is None or lastIdx is None:
        return values[0:0]  # empty slice
    return values[firstIdx:lastIdx+1]


def valueSliceOuter(keys, lowerBoundKey, upperBoundKey, values=None, fallbackBounds=True):
    """
    For a sorted array of keys and an array of corresponding values,
    finds indices i, j such that i is the largest key where keys[i] <= lowerBoundKey and
    j is the lowest key where keys[j] <= upperBoundKey,
    and returns the corresponding slice of values values[i:j+1].
    If such indices do not exists and fallbackBounds==True, the array bounds are used (i.e. 0 or len-1).
    Given the indices i and j, the function returns values[i:j+1].
    This is an outer slice which is the smallest slice that definitely contains two given bounds
    (for fallbackBounds==False).

    :param keys: the sorted array of key values
    :param lowerBoundKey: the key value defining the lower bound
    :param upperBoundKey: the key value defining the upper bound
    :param values: the sorted array of values; if None, use keys
    :return: the corresponding slice of `values`
    """
    if values is None:
        values = keys
    elif len(values) != len(keys):
        raise ValueError("Length of values must match length of keys")
    if lowerBoundKey > upperBoundKey:
        raise ValueError(f"Invalid bounds: {lowerBoundKey} to {upperBoundKey}")
    firstIdx = floorIndex(keys, lowerBoundKey)
    lastIdx = ceilIndex(keys, upperBoundKey)
    if firstIdx is None:
        if fallbackBounds:
            firstIdx = 0
        else:
            raise Exception(f"Bounds would exceed start of array (lowerBoundKey={lowerBoundKey}, lowest key={keys[0]})")
    if lastIdx is None:
        if fallbackBounds:
            lastIdx = len(keys) - 1
        else:
            raise Exception(f"Bounds would exceed end of array (upperBoundKey={upperBoundKey}, largest key={keys[-1]})")
    return values[firstIdx:lastIdx+1]
