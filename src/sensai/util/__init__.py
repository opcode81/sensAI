from typing import Sequence, TypeVar, List

from . import cache

T = TypeVar("T")


def concatSequences(seqs: Sequence[Sequence[T]]) -> List[T]:
    result = []
    for s in seqs:
        result.extend(s)
    return result


def dict2OrderedTuples(d: dict):
    keys = sorted(d.keys())
    values = [d[k] for k in keys]
    return tuple(keys), tuple(values)
