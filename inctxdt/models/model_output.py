from dataclasses import dataclass, field

import torch
from typing import Any, Tuple

from transformers.modeling_outputs import BaseModelOutput


class MetaOutput(type):
    def __call__(cls, *args, **kwargs):
        if kwargs.pop("only_logits", None):
            return kwargs.get("logits", *args)

        # if i have an extra key should i modify the class?
        return super().__call__(*args, **kwargs)


@dataclass
# class ModelOutput(OrderedDict, metaclass=MetaOutput):
class ModelOutput(metaclass=MetaOutput):
    logits: torch.Tensor
    extra: Any = field(default=None, repr=False)

    def __iter__(self):
        return iter(self.__dict__.items())

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())

    def __getitem__(self, k):
        return self.__dict__[k]

    def __setitem__(self, k, v):
        self.__dict__[k] = v

    def keys(self):
        return [k for k, v in self.__dict__.items() if v is not None]

    def items(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}.items()
