from typing import Union, List, Dict, Any, Sequence
import re


def dictString(d):
    return ', '.join([f'{k}={v}' for k, v in d.items()])


def objectRepr(obj, memberNamesOrDict: Union[List[str], Dict[str, Any]]):
    def toString(x):
        if type(x) == dict:
            return "{" + dictString(x) + "}"
        else:
            return str(x)

    if type(memberNamesOrDict) == dict:
        membersDict = memberNamesOrDict
    else:
        membersDict = {m: toString(getattr(obj, m)) for m in memberNamesOrDict}
    return f"{obj.__class__.__name__}[{dictString(membersDict)}]"


def orRegexGroup(allowedNames: Sequence[str]):
    """

    :param allowedNames: strings to include as literals in the regex
    :return: raw string of the type (<name1>| ...|<nameN>), where special characters in the names have been escaped
    """
    allowedNames = [re.escape(name) for name in allowedNames]
    return r"(%s)" % "|".join(allowedNames)
