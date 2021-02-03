from typing import Optional, TypeVar, Sequence

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
