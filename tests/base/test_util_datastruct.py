from sensai.util.datastruct import SortedKeyValuePairs, SortedKeysAndValues


def test_sorted_structures():
    keys = [1, 3, 5, 7, 10]
    values = [str(k) for k in keys]
    sortedPairs = SortedKeyValuePairs(list(zip(keys, values)))
    sortedKV = SortedKeysAndValues(keys, values)

    for s in (sortedPairs, sortedKV):
        assert s.ceilIndex(2) == 1
        assert s.ceilValue(2) == "3"
        assert s.ceilKeyAndValue(2) == (3, "3")
        assert s.floorIndex(8) == 3
        assert s.floorValue(8) == "7"
        assert s.floorKeyAndValue(8) == (7, "7")
        assert s.floorIndex(0) is None
        assert s.floorValue(0) is None
        assert s.ceilIndex(11) is None
        assert s.ceilValue(11) is None
        assert s.closestValue(2.5) == "3"
        assert s.closestKeyAndValue(2.5) == (3, "3")
        assert s.closestIndex(2.5) == 1
        assert s.closestValue(1.1) == "1"
        assert s.closestKeyAndValue(1.1) == (1, "1")
        assert s.closestIndex(1.1) == 0
        assert len(sortedPairs.slice(3, 5, inner=True)) == 2
        assert len(sortedPairs.slice(3, 5, inner=False)) == 2
        assert len(sortedPairs.slice(2, 8, inner=True)) == 3
        assert len(sortedPairs.slice(2, 8, inner=False)) == 5
        assert len(sortedPairs.slice(0, 12, inner=False)) == len(keys)
        assert len(sortedPairs.slice(12, 14, inner=False)) == 1
        assert len(sortedPairs.slice(12, 14, inner=True)) == 0
        assert len(sortedPairs.slice(-1, 0, inner=True)) == 0
        assert len(sortedPairs.slice(-1, 0, inner=False)) == 1

    assert len(sortedKV.valueSliceInner(2, 8)) == 3

    assert len(sortedPairs.valueSlice(2, 8)) == 3
    assert sortedPairs.keyForIndex(3) == 7

    sortedPairs = SortedKeyValuePairs(list(zip(keys, keys)))
    sortedKV = SortedKeysAndValues(keys, keys)
    for s in (sortedPairs, sortedKV):
        assert s.interpolatedValue(2) == 2

