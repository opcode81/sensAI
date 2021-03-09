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
    """
    Provides default implementations for __str__ and __repr__ which contain all attribute names and their values. The
    latter also contains the object id.
    """

    def _toStringClassName(self):
        return type(self).__qualname__

    def _toStringProperties(self, exclude: Union[str, Iterable[str]] = None, **additionalEntries) -> str:
        """
        Creates a string of the class attributes, optionally excluding some and adding others.

        :param exclude: attributes to be excluded
        :param additionalEntries: additional key-value-pairs which are added to the string just like the other attributes
        :return: a string containing all attribute names and values
        """
        if exclude is None:
            exclude = []
        elif type(exclude) == str:
            exclude = [exclude]
        d = {k: v for k, v in self.__dict__.items() if k not in exclude}
        d.update(additionalEntries)
        return dictString(d)

    def _toStringObjectInfo(self) -> str:
        """
        Creates a string containing information on the objects state, which is the name and value of all attributes
        without the attributes that are in the list provided by _toStringExclusions. It is used by the methods __str__
        and __repr__. This method can be overwritten by sub-classes to provide a custom string.

        :return: a string containing all attribute names and values
        """
        return self._toStringProperties(exclude=self._toStringExcludes())

    def _toStringExcludes(self) -> List[str]:
        """
        Returns a list of attribute names to be excluded from __str__ and __repr__. This method can be overwritten by
        sub-classes which can call super to extend this list. This method will only have an effect if _toStringObjectInfo
        is not overwritten by the sub class.

        :return: a list of attribute names
        """
        return []

    def __str__(self):
        return f"{self._toStringClassName()}[{self._toStringObjectInfo()}]"

    def __repr__(self):
        info = f"id={id(self)}"
        propertyInfo = self._toStringObjectInfo()
        if len(propertyInfo) > 0:
            info += ", " + propertyInfo
        return f"{self._toStringClassName()}[{info}]"