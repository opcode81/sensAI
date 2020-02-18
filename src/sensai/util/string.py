from typing import Union, List, Dict, Any


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
