from sensai.util.datastruct import SortedKeyValuePairs, SortedKeysAndValues


def test_sorted_structures():
    keys = [1, 3, 5, 7, 10]
    values = [str(k) for k in keys]
    sortedPairs = SortedKeyValuePairs(list(zip(keys, values)))
    sortedKV = SortedKeysAndValues(keys, values)

    for s in (sortedPairs, sortedKV):
        assert s.ceil_index(2) == 1
        assert s.ceil_value(2) == "3"
        assert s.ceil_key_and_value(2) == (3, "3")
        assert s.floor_index(8) == 3
        assert s.floor_value(8) == "7"
        assert s.floor_key_and_value(8) == (7, "7")
        assert s.floor_index(0) is None
        assert s.floor_value(0) is None
        assert s.ceil_index(11) is None
        assert s.ceil_value(11) is None
        assert s.closest_value(2.5) == "3"
        assert s.closest_key_and_value(2.5) == (3, "3")
        assert s.closest_index(2.5) == 1
        assert s.closest_value(1.1) == "1"
        assert s.closest_key_and_value(1.1) == (1, "1")
        assert s.closest_index(1.1) == 0
        assert len(sortedPairs.slice(3, 5, inner=True)) == 2
        assert len(sortedPairs.slice(3, 5, inner=False)) == 2
        assert len(sortedPairs.slice(2, 8, inner=True)) == 3
        assert len(sortedPairs.slice(2, 8, inner=False)) == 5
        assert len(sortedPairs.slice(0, 12, inner=False)) == len(keys)
        assert len(sortedPairs.slice(12, 14, inner=False)) == 1
        assert len(sortedPairs.slice(12, 14, inner=True)) == 0
        assert len(sortedPairs.slice(-1, 0, inner=True)) == 0
        assert len(sortedPairs.slice(-1, 0, inner=False)) == 1

    assert len(sortedKV.value_slice_inner(2, 8)) == 3

    assert len(sortedPairs.value_slice(2, 8)) == 3
    assert sortedPairs.key_for_index(3) == 7

    sortedPairs = SortedKeyValuePairs(list(zip(keys, keys)))
    sortedKV = SortedKeysAndValues(keys, keys)
    for s in (sortedPairs, sortedKV):
        assert s.interpolated_value(2) == 2

