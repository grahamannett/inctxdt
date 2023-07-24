from dataclasses import asdict

import torch
from tensordict import tensorclass

from inctxdt.episode_data import EpisodeData


@tensorclass
class ObservationsTensorClass:
    observation: torch.Tensor
    achieved_goal: torch.Tensor
    desired_goal: torch.Tensor


def flatten_dataclass(instance):
    def _flatten(value):
        if isinstance(value, dict):
            return {k: _flatten(v) for k, v in value.items()}
        return value

    for key, value in asdict(instance).items():
        setattr(instance, key, _flatten(value))


class TensorclassHelper:
    class Base:
        pass

    @staticmethod
    def from_sample(sample: EpisodeData, name: str = None):
        sample = asdict(sample)

        class Base:
            pass

        if name is not None:
            Base.__name__ = name

        for key, value in sample.items():
            setattr(Base, key, value)

        return tensorclass(Base)
