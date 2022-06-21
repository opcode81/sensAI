import atexit
import logging as lg
from logging import *
import sys
import time
from typing import List, Tuple

import pandas as pd

log = getLogger(__name__)

LOG_DEFAULT_FORMAT = '%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s'


def removeLogHandlers():
    """
    Removes all current log handlers
    """
    logger = getLogger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])


def isLogHandlerActive(handler):
    return handler in getLogger().handlers


def configureLogging(format=LOG_DEFAULT_FORMAT, level=lg.DEBUG):
    removeLogHandlers()
    basicConfig(level=level, format=format, stream=sys.stdout)
    getLogger("matplotlib").setLevel(lg.INFO)
    getLogger("urllib3").setLevel(lg.INFO)
    getLogger("msal").setLevel(lg.INFO)
    pd.set_option('display.max_colwidth', 255)


_fileLoggerPaths: List[str] = []
_isAtExitReportFileLoggerRegistered = False


def _atExitReportFileLogger():
    for path in _fileLoggerPaths:
        print(f"A log file was saved to {path}")


def addFileLogger(path):
    global _isAtExitReportFileLoggerRegistered
    log.info(f"Logging to {path} ...")
    handler = FileHandler(path)
    handler.setFormatter(Formatter(LOG_DEFAULT_FORMAT))
    Logger.root.addHandler(handler)
    _fileLoggerPaths.append(path)
    if not _isAtExitReportFileLoggerRegistered:
        atexit.register(_atExitReportFileLogger)
        _isAtExitReportFileLoggerRegistered = True


class StopWatch:
    """
    Represents a stop watch for timing an execution. Constructing an instance starts the stopwatch.
    """
    def __init__(self):
        self.startTime = time.time()

    def restart(self):
        self.startTime = time.time()

    def getElapsedTimeSecs(self) -> float:
        return time.time() - self.startTime

    def getElapsedTimedelta(self) -> pd.Timedelta:
        return pd.Timedelta(self.getElapsedTimeSecs(), unit="s")

    def getElapsedTimeString(self) -> str:
        secs = self.getElapsedTimeSecs()
        if secs < 60:
            return f"{secs:.3f} seconds"
        else:
            return str(pd.Timedelta(secs, unit="s"))


class StopWatchManager:
    """
    A singleton which manages a pool of named stopwatches, such that executions to be timed by referring to a name only -
    without the need for a limited scope.
    """
    _instance = None

    @classmethod
    def getInstance(cls):
        if cls._instance is None:
            cls._instance = StopWatchManager(42)
        return cls._instance

    def __init__(self, secret):
        if secret != 42:
            raise Exception("Use only the singleton instance via getInstance")
        self._stopWatches = {}

    def start(self, name):
        self._stopWatches[name] = time.time()

    def stop(self, name) -> float:
        """
        :param name: the name of the stopwatch
        :return: the time that has passed in seconds
        """
        timePassedSecs = time.time() - self._stopWatches[name]
        del self._stopWatches[name]
        return timePassedSecs

    def isRunning(self, name):
        return name in self._stopWatches


class LogTime:
    """
    An execution time logger which can be conveniently applied using a with-statement - in order to log the executing time of the respective
    with-block.
    """

    def __init__(self, name, enabled=True, logger: Logger = None):
        """
        :param name: the name of the event whose time is to be logged upon completion as "<name> completed in <time>"
        :param enabled: whether the logging is actually enabled; can be set to False to disable logging without necessitating
            changes to client code
        :param logger: the logger to use; if None, use the logger of LogTime's module
        """
        self.name = name
        self.enabled = enabled
        self.stopwatch = None
        self.logger = logger if logger is not None else log

    def start(self):
        """
        Starts the stopwatch
        """
        self.stopwatch = StopWatch()
        if self.enabled:
            self.logger.info(f"{self.name} starting ...")

    def stop(self):
        """
        Stops the stopwatch and logs the time taken (if enabled)
        """
        if self.stopwatch is not None and self.enabled:
            self.logger.info(f"{self.name} completed in {self.stopwatch.getElapsedTimeString()}")

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def __enter__(self):
        self.start()
        return self