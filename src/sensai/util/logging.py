import time


class StopWatch:
    """
    A simple stopwatch singleton which can be used to determine execution times
    """
    _instance = None

    @classmethod
    def getInstance(cls):
        if cls._instance is None:
            cls._instance = StopWatch(42)
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