from typing import Sequence, TypeVar, List

from . import cache

T = TypeVar("T")


def concatSequences(seqs: Sequence[Sequence[T]]) -> List[T]:
    result = []
    for s in seqs:
        result.extend(s)
    return result