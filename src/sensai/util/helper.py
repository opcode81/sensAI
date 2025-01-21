"""
This module contains various helper functions.
"""
import math
from typing import Any, Sequence, Union, TypeVar, List, Optional, Dict, Container, Iterable, Tuple

T = TypeVar("T")


def count_none(*args: Any) -> int:
    """
    Counts the number of arguments that are None

    :param args: various arguments
    :return: the number of arguments that are None
    """
    c = 0
    for a in args:
        if a is None:
            c += 1
    return c


def count_not_none(*args: Any) -> int:
    """
    Counts the number of arguments that are not None

    :param args: various arguments
    :return: the number of arguments that are not None
    """
    return len(args) - count_none(*args)


def any_none(*args: Any) -> bool:
    """
    :param args: various arguments
    :return: True if any of the arguments are None, False otherwise
    """
    return count_none(*args) > 0


def all_none(*args: Any) -> bool:
    """
    :param args: various arguments
    :return: True if all of the arguments are None, False otherwise
    """
    return count_none(*args) == len(args)


def check_not_nan_dict(d: dict):
    """
    Raises ValueError if any of the values in the given dictionary are NaN, reporting the respective keys

    :param d: a dictionary mapping to floats that are to be checked for NaN
    """
    invalid_keys = [k for k, v in d.items() if math.isnan(v)]
    if len(invalid_keys) > 0:
        raise ValueError(f"Got one or more NaN values: {invalid_keys}")


def contains_any(container: Union[Container, Iterable], items: Sequence) -> bool:
    for item in items:
        if item in container:
            return True
    return False


# noinspection PyUnusedLocal
def mark_used(*args):
    """
    Utility function to mark identifiers as used.
    The function does nothing.

    :param args: pass identifiers that shall be marked as used here
    """
    pass


def flatten_arguments(args: Sequence[Union[T, Sequence[T]]]) -> List[T]:
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


def kwarg_if_not_none(arg_name: str, arg_value: Any) -> Dict[str, Any]:
    """
    Supports the optional passing of a kwarg, returning a non-empty dictionary with the kwarg only
    if the value is not None.

    This can be helpful to improve backward compatibility for cases where a kwarg was added later
    but old implementations weren't updated to have it, such that an exception will be raised only
    if the kwarg is actually used.

    :param arg_name: the argument name
    :param arg_value: the value
    :return: a dictionary containing the kwarg or, if the value is None, an empty dictionary
    """
    if arg_value is None:
        return {}
    else:
        return {arg_name: arg_value}


def flatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary by concatenating nested keys with a separator.

    :param d: the dictionary to flatten
    :param sep: the separator to use in order to join the keys of nested dictionaries
    :return: a flattened dictionary
    """
    def _flatten(d: Dict[str, Any], parent_key: str = '') -> List[Tuple[str, Any]]:
        items: List[Tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key))
            else:
                items.append((new_key, v))
        return items

    return dict(_flatten(d))