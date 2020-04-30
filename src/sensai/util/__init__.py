import logging
import time
from typing import Sequence, TypeVar, List

from . import cache

T = TypeVar("T")
log = logging.getLogger(__name__)


def concatSequences(seqs: Sequence[Sequence[T]]) -> List[T]:
    result = []
    for s in seqs:
        result.extend(s)
    return result


def dict2OrderedTuples(d: dict):
    keys = sorted(d.keys())
    values = [d[k] for k in keys]
    return tuple(keys), tuple(values)


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
