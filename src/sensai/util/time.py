import pandas as pd


def tsNextMonth(ts: pd.Timestamp) -> pd.Timestamp:
    m = ts.month
    if m == 12:
        return ts.replace(year=ts.year+1, month=1)
    else:
        return ts.replace(month=m+1)


def timeOfDay(ts: pd.Timestamp) -> float:
    """
    :param ts: the timestamp
    :return: the time of day as a floating point number in [0, 24)
    """
    return ts.hour + ts.minute / 60


class TimeInterval:
    def __init__(self, start: pd.Timestamp, end: pd.Timestamp):
        self.start = start
        self.end = end

    def contains(self, t: pd.Timestamp):
        return self.start <= t <= self.end

    def overlapsWith(self, other: "TimeInterval") -> bool:
        otherEndsBefore = other.end <= self.start
        otherStartsAfter = other.start >= self.end
        return not (otherEndsBefore or otherStartsAfter)

    def intersection(self, other: "TimeInterval") -> "TimeInterval":
        return TimeInterval(max(self.start, other.start), min(self.end, other.end))

    def timeDelta(self) -> pd.Timedelta:
        return self.end - self.start

    def midTimestamp(self) -> pd.Timestamp:
        midTime: pd.Timestamp = self.start + 0.5 * self.timeDelta()
        return midTime