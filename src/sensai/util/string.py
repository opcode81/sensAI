from typing import Union, List, Dict, Any, Sequence, Iterable
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


class ToStringMixin:
    def _toStringClassName(self):
        return type(self).__qualname__

    def _toStringProperties(self, exclude: Union[str, Iterable[str]] = None, **additionalEntries) -> str:
        if exclude is None:
            exclude = []
        elif type(exclude) == str:
            exclude = [exclude]
        d = {k: v for k, v in self.__dict__.items() if k not in exclude}
        d.update(additionalEntries)
        return dictString(d)

    def _toStringObjectInfo(self) -> str:
        return self._toStringProperties()

    def __str__(self):
        return f"{self._toStringClassName()}[{self._toStringObjectInfo()}]"

    def __repr__(self):
        info = f"id={id(self)}"
        propertyInfo = self._toStringObjectInfo()
        if len(propertyInfo) > 0:
            info += ", " + propertyInfo
        return f"{self._toStringClassName()}[{info}]"