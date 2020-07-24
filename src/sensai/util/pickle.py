import logging
import os
import pickle
from typing import List

from ..base.interfaces import LoadSaveInterface

log = logging.getLogger(__name__)


class PickleFailureDebugger:
    """
    A collection of methods for testing whether objects can be pickled and logging useful infos in case they cannot
    """

    enabled = False  # global flag controlling the behaviour of logFailureIfEnabled

    @classmethod
    def _debugFailure(cls, obj, path, failures, handledObjectIds):
        if id(obj) in handledObjectIds:
            return
        handledObjectIds.add(id(obj))

        try:
            pickle.dumps(obj)
        except:
            # determine dictionary of children to investigate (if any)
            if hasattr(obj, '__dict__'):  # Because of strange behaviour of getstate, here try-except is used instead of if-else
                try:  # Because of strange behaviour of getattr(_, '__getstate__'), we here use try-except
                    d = obj.__getstate__()
                    if type(d) != dict:
                        d = {"state": d}
                except:
                    d = obj.__dict__
            elif type(obj) == dict:
                d = obj
            elif type(obj) in (list, tuple, set):
                d = dict(enumerate(obj))
            else:
                d = {}

            # recursively test children
            haveFailedChild = False
            for key, child in d.items():
                childPath = list(path) + [f"{key}[{child.__class__.__name__}]"]
                haveFailedChild = cls._debugFailure(child, childPath, failures, handledObjectIds) or haveFailedChild

            if not haveFailedChild:
                failures.append(path)

            return True
        else:
            return False

    @classmethod
    def debugFailure(cls, obj) -> List[str]:
        """
        Recursively tries to pickle the given object and returns a list of failed paths

        :param obj: the object for which to recursively test pickling
        :return: a list of object paths that failed to pickle
        """
        handledObjectIds = set()
        failures = []
        cls._debugFailure(obj, [obj.__class__.__name__], failures, handledObjectIds)
        return [".".join(l) for l in failures]

    @classmethod
    def logFailureIfEnabled(cls, obj, contextInfo: str = None):
        """
        If the class flag 'enabled' is set to true, the pickling of the given object is
        recursively tested and the results are logged at error level if there are problems and
        info level otherwise.
        If the flag is disabled, no action is taken.

        :param obj: the object for which to recursively test pickling
        :param contextInfo: optional additional string to be included in the log message
        """
        if cls.enabled:
            failures = cls.debugFailure(obj)
            prefix = f"Picklability analysis for {obj}"
            if contextInfo is not None:
                prefix += " (context: %s)" % contextInfo
            if len(failures) > 0:
                log.error(f"{prefix}: pickling would result in failures due to: {failures}")
            else:
                log.info(f"{prefix}: is picklable")


# Todo or not todo: Use this in vector model
class PickleSerializingMixin(LoadSaveInterface):
    def save(self, path: str):
        """
        Saves the instance as pickle
        """
        basedir = os.path.dirname(path)
        os.makedirs(basedir, exist_ok=True)
        log.info(f"Saving instance of {self.__class__.__name__} to {path}")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Loads a class instance from pickle
        :param path:
        :return: instance of the present class
        """
        log.info(f"Loading instance of {cls} from {path}")
        with open(path, 'rb') as f:
            result = pickle.load(f)
        if not isinstance(result, cls):
            raise Exception(f"Excepted instance of {cls}, instead got: {result.__class__.__name__}")
        return result
