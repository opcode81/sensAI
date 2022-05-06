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
