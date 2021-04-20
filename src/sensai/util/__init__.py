import logging
import time
from typing import Sequence, TypeVar, List, Union

from . import cache
from . import cache_mysql

T = TypeVar("T")
log = logging.getLogger(__name__)


def countNone(*args):
    c = 0
    for a in args:
        if a is None:
            c += 1
    return c


def anyNone(*args):
    return countNone(*args) > 0


def allNone(*args):
    return countNone(*args) == len(args)


def concatSequences(seqs: Sequence[Sequence[T]]) -> List[T]:
    result = []
    for s in seqs:
        result.extend(s)
    return result


def dict2OrderedTuples(d: dict):
    keys = sorted(d.keys())
    values = [d[k] for k in keys]
    return tuple(keys), tuple(values)


def flattenArguments(args: Sequence[Union[T, Sequence[T]]]) -> List[T]:
    """
    Main use case is to support both interfaces of the type f(T1, T2, ...) and f([T1, T2, ...]) simultaneously.
    It is assumed that if the latter form is passed, the arguments are either in a list or a tuple. Moreover,
    T cannot be a tuple or a list itself.

    Overall this function is not all too safe and one should be aware of what one is doing when using it
    """
    result = []
    for arg in args:
        if isinstance(arg, list) or isinstance(arg, tuple):
            result.extend(arg)
        else:
            result.append(arg)
    return result


def markUsed(*args):
    """
    Utility function to mark identifiers as used.
    The function does nothing.

    :param args: pass identifiers that shall be marked as used here
    """
    pass


class LogTime:
    def __init__(self, name):
        self.name = name
        self.startTime = None

    def start(self):
        self.startTime = time.time()

    def stop(self):
        log.info(f"{self.name} completed in {time.time()-self.startTime} seconds")

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def __enter__(self):
        return self.start()
