from dataclasses import dataclass, field

import torch
from typing import Any, Tuple
from collections import UserDict

from transformers.modeling_outputs import BaseModelOutput


@dataclass
class ModelOutput(UserDict):
    logits: torch.Tensor
    extra: Any = field(default=None, repr=False)

    def __iter__(self):
        return iter((self.logits, self.extra))

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())
