from typing import Sequence, TypeVar, List

from . import cache

T = TypeVar("T")


def concatSequences(seqs: Sequence[Sequence[T]]) -> List[T]:
    result = []
    for s in seqs:
        result.extend(s)
    return result


class PicklableFunction:
    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        actualArgs = list(self.args)
        actualArgs.extend(args)
        actualKwargs = dict(self.kwargs)
        actualKwargs.update(kwargs)
        return self.fn(*actualArgs, **actualKwargs)
