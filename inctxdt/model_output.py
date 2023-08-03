from dataclasses import dataclass, field

import torch
from typing import Any, Tuple
from collections import UserDict

from transformers.modeling_outputs import BaseModelOutput


@dataclass
class ModelOutput(UserDict):
    logits: torch.Tensor
    extra: Any = field(default=None, repr=False)

    def __new__(cls, *args, **kwargs):
        """all argument to be passed so we can only send out logits if needed
        for instance testing with another person eval function

        Returns:
            _type_: _description_
        """
        if kwargs.get("only_logits", False):
            return kwargs.get("logits", *args)

        return super().__new__(cls)

    def __iter__(self):
        return iter(self.__dict__.items())

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())
