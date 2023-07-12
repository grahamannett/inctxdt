from typing import Optional

from dataclasses import dataclass
import numpy as np


class EpisodeDataConfig:
    frozen: bool = False


@dataclass(frozen=EpisodeDataConfig.frozen)
class EpisodeData:
    """Contains the datasets data for a single episode.

    This is the object returned by :class:`minari.MinariDataset.sample_episodes`.
    """

    id: int
    seed: Optional[int]
    total_timesteps: int
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    returns_to_go: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray
    timesteps: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.timesteps is None:
            self.timesteps = np.arange(self.observations.shape[0])

    def __repr__(self) -> str:
        return (
            "EpisodeData("
            f"id={repr(self.id)}, "
            f"seed={repr(self.seed)}, "
            f"total_timesteps={self.total_timesteps}, "
            f"observations={EpisodeData._repr_space_values(self.observations)}, "
            f"actions={EpisodeData._repr_space_values(self.actions)}, "
            f"rewards=ndarray of {len(self.rewards)} floats, "
            f"returns_to_go=ndarray of {len(self.returns_to_go)} floats, "
            f"terminations=ndarray of {len(self.terminations)} bools, "
            f"truncations=ndarray of {len(self.truncations)} bools"
            ")"
        )

    @staticmethod
    def _repr_space_values(value):
        if isinstance(value, np.ndarray):
            return f"ndarray of shape {value.shape} and dtype {value.dtype}"
        elif isinstance(value, dict):
            reprs = [f"{k}: {EpisodeData._repr_space_values(v)}" for k, v in value.items()]
            dict_repr = ", ".join(reprs)
            return "{" + dict_repr + "}"
        elif isinstance(value, tuple):
            reprs = [EpisodeData._repr_space_values(v) for v in value]
            values_repr = ", ".join(reprs)
            return "(" + values_repr + ")"
        else:
            return repr(value)


def make(frozen: bool = EpisodeDataConfig.frozen):
    @dataclass(frozen=frozen)
    class EpisodeData_(EpisodeData):
        pass

    return EpisodeData_
