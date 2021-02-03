import logging
import time
from typing import Callable

log = logging.getLogger(__name__)


def timed(method: Callable):
    """
    Decorator for execution timing
    """
    def timedExecution(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()
        log.info(f"Finished execution of {method.__name__} in {end-start:.2f}s")
        return result
    return timedExecution
