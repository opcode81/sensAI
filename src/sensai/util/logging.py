import time


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
        :param name: the name of the time
        :return: the time that has passed in seconds
        """
        timePassedSecs = time.time() - self._stopWatches[name]
        del self._stopWatches[name]
        return timePassedSecs

    def isRunning(self, name):
        return name in self._stopWatches