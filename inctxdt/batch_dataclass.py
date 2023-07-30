from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


import torch
from inctxdt.batch import Batch
from inctxdt.episode_data import EpisodeData


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
