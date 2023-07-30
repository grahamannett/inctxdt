from typing import List, Optional

import torch
from dataclasses import dataclass, fields
import numpy as np


class EpisodeDataConfig:
    frozen: bool = False


@dataclass
class ObservationData:
    observation: np.ndarray
    achieved_goal: np.ndarray
    desired_goal: np.ndarray


@dataclass(frozen=EpisodeDataConfig.frozen)
class EpisodeData:
    """Contains the datasets data for a single episode.

    This is the object returned by :class:`minari.MinariDataset.sample_episodes`.
    """

    id: int | List[int]
    seed: Optional[int] | List[Optional[int]]
    total_timesteps: int | List[int]
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    returns_to_go: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray
    timesteps: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    env_name: Optional[str] = None

    def __post_init__(self):
        if self.timesteps is None:
            self.timesteps = np.arange(max(self.total_timesteps, len(self.observations)))

    def __repr__(self) -> str:
        return (
            "EpisodeData("
            f"env_name={repr(self.env_name)},"
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

    @classmethod
    def combine(cls, episodes: List["EpisodeData"], skip_fields: List[str] = ["env_name"]) -> "EpisodeData":
        if len(env_names := set([ep.env_name for ep in episodes])) > 1:
            raise ValueError(f"Cannot combine episodes from different environments: {env_names}")

        out_ep = {"env_name": env_names.pop()}

        for field in fields(episodes[0]):
            if field.name in skip_fields:
                continue

            comb_field = [getattr(ep, field.name) for ep in episodes]
            if isinstance(comb_field[0], np.ndarray):
                comb_field = np.concatenate(comb_field, axis=0)

            out_ep[field.name] = comb_field

        return cls(**out_ep)


def make(frozen: bool = EpisodeDataConfig.frozen):
    class EpisodeData(EpisodeData):
        pass

    return dataclass(frozen=frozen)(EpisodeData)
