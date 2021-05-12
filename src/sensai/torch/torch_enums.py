from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, Union

from torch.nn import functional as F


@dataclass
class _ActivationFunction:
    name: str
    fn: Optional[Callable]


class ActivationFunction(Enum):
    NONE = _ActivationFunction("none", None)
    SIGMOID = _ActivationFunction("sigmoid", F.sigmoid)
    RELU = _ActivationFunction("relu", F.relu)
    TANH = _ActivationFunction("tanh", F.tanh)
    LOG_SOFTMAX = _ActivationFunction("log_softmax", F.log_softmax)
    SOFTMAX = _ActivationFunction("softmax", F.softmax)

    @classmethod
    def fromName(cls, name) -> "ActivationFunction":
        for item in cls:
            if item.value.name == name:
                return item
        raise ValueError(f"No function found for name '{name}'")

    def getTorchFunction(self) -> Callable:
        return self.value.fn

    def getName(self) -> str:
        return self.value.name

    @classmethod
    def torchFunctionFromAny(cls, f: Union[str, "ActivationFunction", Callable]) -> Callable:
        """
        Gets the torch activation for the given argument

        :param f: either an instance of ActivationFunction, the name of a function from torch.nn.functional or an actual function
        :return: a function that can be applied to tensors
        """
        if isinstance(f, str):
            return getattr(F, f)
        elif isinstance(f, ActivationFunction):
            return f.getTorchFunction()
        elif callable(f):
            return f
        else:
            raise ValueError()


class ClassificationOutputMode(Enum):
    PROBABILITIES = "probabilities"
    LOG_PROBABILITIES = "log_probabilities"
    UNNORMALISED_LOG_PROBABILITIES = "unnormalised_log_probabilities"

    @classmethod
    def forActivationFn(cls, fn: Optional[Callable]):
        if fn is None:
            return cls.UNNORMALISED_LOG_PROBABILITIES
        name = fn.__name__
        if name in ("sigmoid", "relu"):
            raise ValueError(f"The activation function {fn} is not suitable as an output activation function for classifcation")
        elif name in ("log_softmax",):
            return cls.LOG_PROBABILITIES
        elif name in ("softmax",):
            return cls.PROBABILITIES
        else:
            raise ValueError(f"Unhandled function {fn}")