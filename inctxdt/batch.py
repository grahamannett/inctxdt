from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

from tensordict import TensorDict, tensorclass

import torch

from inctxdt.episode_data import EpisodeData

# helper functions for padding and stacknig


def from_eps_with_pad(eps, attr: str, batch_first: bool = True) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence([torch.from_numpy(getattr(x, attr)) for x in eps], batch_first=batch_first)


def from_eps(eps: List[EpisodeData], attr: str) -> torch.Tensor:
    return torch.tensor([getattr(x, attr) for x in eps])


# class PadWrapper:
#     def __init__(self, episode_list: "EpisodeList"):
#         self.episode_list = episode_list
#         self._pad_kwargs = {"batch_first": self.episode_list.batch_first}

#     def __getattr__(self, attr):
#         return from_eps_with_pad(self.episode_list.eps, attr, **self._pad_kwargs)


# class ListWrapper:
#     def __init__(self, episode_list: "EpisodeList"):
#         self.episode_list = episode_list

#     def __getattr__(self, attr):
#         return [getattr(ep, attr) for ep in self.episode_list.eps]


class EpisodeList(list):
    def __init__(self, eps: List[EpisodeData], batch_first: bool = True):
        super().__init__(eps)
        self.batch_first = batch_first

    def __getattr__(self, attr):
        return FieldList([getattr(ep, attr) for ep in self], batch_first=self.batch_first)


class FieldList(list):
    def __init__(self, arr: list, batch_first: bool = True):
        super().__init__(arr)
        self.batch_first = batch_first

    @property
    def tensor(self):
        return torch.tensor(self)

    @property
    def pad_tensor(self):
        return torch.nn.utils.rnn.pad_sequence([torch.from_numpy(a) for a in self], batch_first=self.batch_first)

    def tensor_(self, *args, **kwargs):
        return torch.tensor(self, *args, **kwargs)

    def pad_tensor_(self, **kwargs):
        return torch.nn.utils.rnn.pad_sequence([torch.from_numpy(a) for a in self], **kwargs)


@tensorclass
class Batch:
    id: torch.Tensor
    total_timesteps: torch.Tensor
    observations: torch.Tensor | TensorDict
    actions: torch.Tensor
    rewards: torch.Tensor
    returns_to_go: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    seed: Optional[torch.Tensor] = None  # out of order with EpisodeData
    timesteps: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    env_name: Optional[List[str]] = None

    @classmethod
    def with_episode_list(cls, episodes: List[EpisodeData], batch_first: bool = True):
        eps = EpisodeList(episodes, batch_first=batch_first)
        return cls(
            id=eps.id,
            total_timesteps=eps.total_timesteps.tensor,
            observations=eps.observations.pad_tensor,
            actions=eps.actions.pad_tensor,
            rewards=eps.rewards.pad_tensor,
            returns_to_go=eps.returns_to_go.pad_tensor,
            terminations=eps.terminations.pad_tensor,
            truncations=eps.truncations.pad_tensor,
            timesteps=eps.timesteps.pad_tensor,
            mask=eps.mask.pad_tensor,
            seed=eps.seed,
            env_name=eps.env_name,
            batch_size=[len(eps)],
        )

    @classmethod
    def with_from_eps(cls, eps: List[EpisodeData], batch_first: bool = True):
        return cls(
            id=from_eps(eps, "id"),
            total_timesteps=from_eps(eps, "total_timesteps"),
            observations=from_eps_with_pad(eps, "observations", batch_first=batch_first),
            actions=from_eps_with_pad(eps, "actions", batch_first=batch_first),
            rewards=from_eps_with_pad(eps, "rewards", batch_first=batch_first),
            returns_to_go=from_eps_with_pad(eps, "returns_to_go", batch_first=batch_first),
            terminations=from_eps_with_pad(eps, "terminations", batch_first=batch_first),
            truncations=from_eps_with_pad(eps, "truncations", batch_first=batch_first),
            timesteps=from_eps_with_pad(eps, "timesteps", batch_first=batch_first),
            mask=from_eps_with_pad(eps, "mask", batch_first=batch_first),
            seed=from_eps(eps, "seed"),
            env_name=[x.env_name for x in eps],
            batch_size=[len(eps)],
        )


class Collate:
    def __init__(self, device: str = "cpu", batch_first: bool = True, return_fn: Callable = Batch.with_episode_list):
        self.device = device
        self.batch_first = batch_first
        self.return_fn = return_fn

    def __call__(self, eps: List[EpisodeData]):
        return self.return_fn(eps, batch_first=self.batch_first).to(self.device)


# these do not use tensordict
def from_eps_with_pad_(eps, attr, dtype, batch_first, device):
    return torch.nn.utils.rnn.pad_sequence([torch.as_tensor(getattr(x, attr), dtype=dtype) for x in eps], batch_first=batch_first).to(
        device
    )


def from_eps_(eps, attr, dtype, device):
    return torch.tensor([getattr(x, attr) for x in eps], dtype=dtype, device=device)


class SamplesDataclass:
    def to(self, device: str):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)
        return self

    def asdict(self) -> Dict[str, torch.Tensor]:
        return asdict(self)


@dataclass
class BatchDataclass(SamplesDataclass):
    id: torch.Tensor
    total_timesteps: torch.Tensor
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    returns_to_go: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    seed: Optional[torch.Tensor] = None  # out of order with EpisodeData
    timesteps: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    env_name: Optional[List[str]] = None

    @classmethod
    def collate_fn(cls, episodes: List[EpisodeData], device: str = "cpu", batch_first: bool = True) -> "Batch":
        return cls(
            id=from_eps_(episodes, "id", torch.int, device),
            seed=from_eps_(episodes, "seed", torch.int, device) if episodes[0].seed else None,
            total_timesteps=from_eps_(episodes, "total_timesteps", int, device),
            observations=from_eps_with_pad_(episodes, "observations", torch.float32, batch_first, device),
            actions=from_eps_with_pad_(episodes, "actions", torch.float32, batch_first, device),
            rewards=from_eps_with_pad_(episodes, "rewards", torch.float32, batch_first, device),
            returns_to_go=from_eps_with_pad_(episodes, "returns_to_go", torch.float32, batch_first, device),
            terminations=from_eps_with_pad_(episodes, "terminations", torch.int, batch_first, device),
            truncations=from_eps_with_pad_(episodes, "truncations", torch.int, batch_first, device),
            timesteps=from_eps_with_pad_(episodes, "timesteps", torch.int, batch_first, device),
            mask=from_eps_with_pad_(episodes, "mask", torch.int, batch_first, device),
            env_name=[x.env_name for x in episodes],
        )

    @classmethod
    def make_collate_fn(cls, device: str = None, batch_first: bool = True):
        def collate_fn(episodes: List[EpisodeData]) -> "Batch":
            return cls.collate_fn(episodes, device=device, batch_first=batch_first)

        return collate_fn
