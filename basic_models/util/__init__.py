from typing import Sequence, TypeVar, List, Dict, Any

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


#TODO: this is obsolete, right?
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
