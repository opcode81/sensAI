

def stringRepr(object, memberNames):
    def toString(x):
        if type(x) == dict:
            return "{" + ", ".join(f"{k}={str(v)}" for k, v in x.items()) + "}"
        else:
            return str(x)

    membersDict = {m: toString(getattr(object, m)) for m in memberNames}
    return f"{object.__class__.__name__}[{', '.join([f'{k}={v}' for k, v in membersDict.items()])}]"
