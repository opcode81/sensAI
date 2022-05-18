from typing import Union, List, Dict, Any, Sequence, Iterable, Optional, Mapping
import re


reCommaWhitespacePotentiallyBreaks = re.compile(r",\s+")


def dictString(d: Mapping, brackets: Optional[str] = None):
    s = ', '.join([f'{k}={toString(v)}' for k, v in d.items()])
    if brackets is not None:
        return brackets[:1] + s + brackets[-1:]
    else:
        return s


def listString(l: Iterable[Any], brackets="[]", quote: Optional[str] = None):
    def item(x):
        x = toString(x)
        if quote is not None:
            return quote + x + quote
        else:
            return x

    return brackets[:1] + ", ".join((item(x) for x in l)) + brackets[-1:]


def toString(x):
    if type(x) == list:
        return listString(x)
    elif type(x) == tuple:
        return listString(x, brackets="()")
    elif type(x) == dict:
        return dictString(x, brackets="{}")
    else:
        s = str(x)
        s = reCommaWhitespacePotentiallyBreaks.sub(", ", s)  # remove any unwanted line breaks and indentation after commas (as generated, for example, by sklearn objects)
        return s


def objectRepr(obj, memberNamesOrDict: Union[List[str], Dict[str, Any]]):
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
    _TOSTRING_INCLUDE_ALL = "__all__"

    def _toStringClassName(self):
        return type(self).__qualname__

    def _toStringProperties(self, exclude: Optional[Union[str, Iterable[str]]] = None, include: Optional[Union[str, Iterable[str]]] = None,
            excludeExceptions: Optional[List[str]] = None, includeForced: Optional[List[str]] = None,
            additionalEntries: Dict[str, Any] = None) -> str:
        """
        Creates a string of the class attributes, with optional exclusions/inclusions/additions.
        Exclusions take precedence over inclusions.

        :param exclude: attributes to be excluded
        :param include: attributes to be included; if None/empty, include all that are not excluded
        :param additionalEntries: additional key-value-pairs which are added to the string just like the other attributes
        :return: a string containing attribute names and values
        """
        def mklist(x):
            if x is None:
                return []
            if type(x) == str:
                return [x]
            return x

        exclude = mklist(exclude)
        include = mklist(include)
        includeForced = mklist(includeForced)
        excludeExceptions = mklist(excludeExceptions)

        def isExcluded(k):
            if k in includeForced or k in excludeExceptions:
                return False
            if k in exclude:
                return True
            if self._toStringExcludePrivate():
                isPrivate = k.startswith("_")
                return isPrivate
            else:
                return False

        # determine relevant attribute dictionary
        if len(include) == 1 and include[0] == self._TOSTRING_INCLUDE_ALL:  # exclude semantics (include everything by default)
            attributeDict = self.__dict__
        else:  # include semantics (include only inclusions)
            attributeDict = {k: getattr(self, k) for k in set(include + includeForced) if hasattr(self, k) and k != self._TOSTRING_INCLUDE_ALL}

        # apply exclusions and remove underscores from attribute names
        d = {k.strip("_"): v for k, v in attributeDict.items() if not isExcluded(k)}

        if additionalEntries is not None:
            d.update(additionalEntries)

        return dictString(d)

    def _toStringObjectInfo(self) -> str:
        """
        Creates a string containing information on the object instance which is to appear between the square brackets in the string
        representation, i.e. if the class name is Foo, then it is the asterisk in "Foo[*]".
        By default will make use of all the exclusions/inclusions that are specified by other member functions.
        This method can be overwritten by sub-classes to provide a custom string.

        :return: a string containing the desired content
        """
        return self._toStringProperties(exclude=self._toStringExcludes(), include=self._toStringIncludes(),
            excludeExceptions=self._toStringExcludeExceptions(), includeForced=self._toStringIncludesForced(),
            additionalEntries=self._toStringAdditionalEntries())

    def _toStringExcludes(self) -> List[str]:
        """
        Makes the string representation exclude the returned attributes.
        Returns a list of attribute names to be excluded from __str__ and __repr__. This method can be overwritten by
        sub-classes which can call super and extend the list returned.
        This method will only have no effect if _toStringObjectInfo is overridden to not use its result.

        :return: a list of attribute names
        """
        return []

    def _toStringIncludes(self) -> List[str]:
        """
        Makes the string representation include only the returned attributes (i.e. introduces inclusion semantics);
        By default, the list contains only a marker element, which is interpreted as "all attributes included".

        This method can be overridden by sub-classes, which can call super in order to extend the list.
        If a list containing the aforementioned marker element (which stands for all attributes) is extended, the marker element will be ignored,
        and only the user-added elements will be considered as included.

        Note: To add an included attribute in a sub-class, regardless of any super-classes using exclusion or inclusion semantics,
        use _toStringIncludesForced instead.

        This method will only have no effect if _toStringObjectInfo is overridden to not use its result.

        :return: a list of attribute names to be included in the string representation
        """
        return [self._TOSTRING_INCLUDE_ALL]

    def _toStringIncludesForced(self) -> List[str]:
        """
        Defines a list of attribute names that are required to be present in the string representation, regardless of the
        instance using include semantics or exclude semantics, thus facilitating added inclusions in sub-classes.

        :return: a list of attribute names
        """
        return []

    def _toStringAdditionalEntries(self) -> Dict[str, Any]:
        return {}

    def _toStringExcludePrivate(self) -> bool:
        """
        :return: whether to exclude properties that are private, i.e. start with an underscore; explicitly included attributes
            will still be considered
        """
        return False

    def _toStringExcludeExceptions(self) -> List[str]:
        """
        Defines attribute names which should not be excluded even though other rules (e.g. the exclusion of private members
        via _toStringExcludePrivate) would otherwise exclude them.

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


def prettyStringRepr(s: Any, initialIndentationLevel=0, indentationString="    "):
    """
    Creates a pretty string representation (using indentations) from the given object/string representation (as generated, for example, via
    ToStringMixin). An indentation level is added for every opening bracket.

    :param s: an object or object string representation
    :param initialIndentationLevel: the initial indentation level
    :param indentationString: the string which corresponds to a single indentation level
    :return: a reformatted version of the input string with added indentations and line break
    """
    if type(s) != str:
        s = str(s)
    indent = initialIndentationLevel
    result = indentationString * indent
    i = 0

    def nl():
        nonlocal result
        result += "\n" + (indentationString * indent)

    def take(cnt=1):
        nonlocal result, i
        result += s[i:i+cnt]
        i += cnt

    def findMatching(j):
        start = j
        op = s[j]
        cl = {"[": "]", "(": ")", "'": "'"}[s[j]]
        isBracket = cl != s[j]
        stack = 0
        while j < len(s):
            if s[j] == op and (isBracket or j == start):
                stack += 1
            elif s[j] == cl:
                stack -= 1
            if stack == 0:
                return j
            j += 1
        return None

    brackets = "[("
    quotes = "'"
    while i < len(s):
        isBracket = s[i] in brackets
        isQuote = s[i] in quotes
        if isBracket or isQuote:
            iMatch = findMatching(i)
            takeFullMatchWithoutBreak = False
            if iMatch is not None:
                k = iMatch + 1
                takeFullMatchWithoutBreak = not isBracket or (k-i <= 60 and not("=" in s[i:k] and "," in s[i:k]))
                if takeFullMatchWithoutBreak:
                    take(k-i)
            if not takeFullMatchWithoutBreak:
                take(1)
                indent += 1
                nl()
        elif s[i] in "])":
            take(1)
            indent -= 1
        elif s[i:i+2] == ", ":
            take(2)
            nl()
        else:
            take(1)

    return result