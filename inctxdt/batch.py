from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

from tensordict import TensorDict, tensorclass

import torch

from inctxdt.episode_data import EpisodeData


def pad_eps_seq(eps, attr: str, batch_first: bool = True) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence([torch.from_numpy(getattr(x, attr)) for x in eps], batch_first=batch_first)


def eps_seq(eps, attr: str) -> torch.Tensor:
    return torch.tensor([getattr(x, attr) for x in eps])


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
    def from_episode_data(cls, eps: List[EpisodeData], batch_first: bool = True):
        return cls(
            id=eps_seq(eps, "id"),
            total_timesteps=eps_seq(eps, "total_timesteps"),
            observations=pad_eps_seq(eps, "observations", batch_first=batch_first),
            actions=pad_eps_seq(eps, "actions", batch_first=batch_first),
            rewards=pad_eps_seq(eps, "rewards", batch_first=batch_first),
            returns_to_go=pad_eps_seq(eps, "returns_to_go", batch_first=batch_first),
            terminations=pad_eps_seq(eps, "terminations", batch_first=batch_first),
            truncations=pad_eps_seq(eps, "truncations", batch_first=batch_first),
            timesteps=pad_eps_seq(eps, "timesteps", batch_first=batch_first),
            mask=pad_eps_seq(eps, "mask", batch_first=batch_first),
            seed=eps_seq(eps, "seed"),
            env_name=[x.env_name for x in eps],
            batch_size=[len(eps)],
        )


class Collate:
    def __init__(self, device: str = "cpu", batch_first: bool = True, return_fn: Callable = Batch.from_episode_data):
        self.device = device
        self.batch_first = batch_first
        self.return_fn = return_fn

    def __call__(self, eps: List[EpisodeData]):
        return self.return_fn(eps, batch_first=self.batch_first).to(self.device)

        # return Batch(
        #     id=eps_seq(eps, "id"),
        #     total_timesteps=eps_seq(eps, "total_timesteps"),
        #     observations=pad_eps_seq(eps, "observations", batch_first=self.batch_first),
        #     actions=pad_eps_seq(eps, "actions", batch_first=self.batch_first),
        #     rewards=pad_eps_seq(eps, "rewards", batch_first=self.batch_first),
        #     returns_to_go=pad_eps_seq(eps, "returns_to_go", batch_first=self.batch_first),
        #     terminations=pad_eps_seq(eps, "terminations", batch_first=self.batch_first),
        #     truncations=pad_eps_seq(eps, "truncations", batch_first=self.batch_first),
        #     timesteps=pad_eps_seq(eps, "timesteps", batch_first=self.batch_first),
        #     mask=pad_eps_seq(eps, "mask", batch_first=self.batch_first),
        #     seed=eps_seq(eps, "seed"),
        #     env_name=[x.env_name for x in eps],
        #     batch_size=[len(eps)],
        # )


# these do not use tensordict
def pad_eps_seq_(eps, attr, dtype, batch_first, device):
    return torch.nn.utils.rnn.pad_sequence([torch.as_tensor(getattr(x, attr), dtype=dtype) for x in eps], batch_first=batch_first).to(
        device
    )


def eps_seq_(eps, attr, dtype, device):
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
            id=eps_seq_(episodes, "id", torch.int, device),
            seed=eps_seq_(episodes, "seed", torch.int, device) if episodes[0].seed else None,
            total_timesteps=eps_seq_(episodes, "total_timesteps", int, device),
            observations=pad_eps_seq_(episodes, "observations", torch.float32, batch_first, device),
            actions=pad_eps_seq_(episodes, "actions", torch.float32, batch_first, device),
            rewards=pad_eps_seq_(episodes, "rewards", torch.float32, batch_first, device),
            returns_to_go=pad_eps_seq_(episodes, "returns_to_go", torch.float32, batch_first, device),
            terminations=pad_eps_seq_(episodes, "terminations", torch.int, batch_first, device),
            truncations=pad_eps_seq_(episodes, "truncations", torch.int, batch_first, device),
            timesteps=pad_eps_seq_(episodes, "timesteps", torch.int, batch_first, device),
            mask=pad_eps_seq_(episodes, "mask", torch.int, batch_first, device),
            env_name=[x.env_name for x in episodes],
        )

    @classmethod
    def make_collate_fn(cls, device: str = None, batch_first: bool = True):
        def collate_fn(episodes: List[EpisodeData]) -> "Batch":
            return cls.collate_fn(episodes, device=device, batch_first=batch_first)

        return collate_fn
