from typing import Callable, Dict, List, Optional

import torch
from tensordict import TensorDict, tensorclass

from inctxdt.episode_data import EpisodeData


# types
class FieldList(list):
    """field list is a the field from all samples before they are stacked"""

    def __init__(self, arr: list, batch_first: bool = True):
        super().__init__(arr)
        self.batch_first = batch_first

    @property
    def tensor(self):
        if (len(self) > 0) and (self[0] is None):
            return

        return torch.tensor(self)

    @property
    def pad_tensor(self):
        if (len(self) > 0) and (self[0] is None):
            return

        return torch.nn.utils.rnn.pad_sequence([torch.from_numpy(a) for a in self], batch_first=self.batch_first)

    # if you need additional args on them you can do this:
    def tensor_(self, *args, **kwargs):
        return torch.tensor(self, *args, **kwargs)

    def pad_tensor_(self, **kwargs):
        return torch.nn.utils.rnn.pad_sequence([torch.from_numpy(a) for a in self], **kwargs)


class EpisodeList(list):
    """EpisodeList is a list of episodes before they are stacked.  The point of this class is basically to make it marginally easier/more user friendly when doing something with a list of episodes"""

    def __init__(self, eps: List[EpisodeData], batch_first: bool = True):
        super().__init__(eps)
        self.batch_first = batch_first

    def __getattr__(self, attr):
        return FieldList([getattr(ep, attr) for ep in self], batch_first=self.batch_first)

    def field(self, field: str, _default=None):
        return [getattr(ep, field, _default) for ep in self]


@tensorclass
class Batch:
    states: torch.Tensor | TensorDict
    actions: torch.Tensor
    total_timesteps: Optional[torch.Tensor] = None

    rewards: Optional[torch.Tensor] = None
    returns_to_go: Optional[torch.Tensor] = None
    terminations: Optional[torch.Tensor] = None
    truncations: Optional[torch.Tensor] = None
    seed: Optional[torch.Tensor] = None  # out of order with EpisodeData
    timesteps: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    id: Optional[torch.Tensor] = None
    env_name: Optional[List[str]] = None

    def make_padding_mask(self):
        return ~self.mask.to(torch.bool)


class Collate:
    def __init__(
        self,
        device: str = None,
        batch_first: bool = True,
        return_fn: Callable[[List[EpisodeData]], Batch] = None,
    ):
        self.device = device
        self.batch_first = batch_first
        self.return_fn = return_fn or self.default_return_fn

    def __call__(self, eps: List[EpisodeData]) -> Batch:
        return self.return_fn(eps)

    def default_return_fn(self, episode_list: List[EpisodeData]) -> Batch:
        eps = EpisodeList(episode_list, batch_first=self.batch_first)
        batch = Batch(
            states=eps.states.pad_tensor,
            actions=eps.actions.pad_tensor,
            total_timesteps=eps.total_timesteps.tensor,
            rewards=eps.rewards.pad_tensor,
            returns_to_go=eps.returns_to_go.pad_tensor,
            terminations=eps.terminations.pad_tensor,
            truncations=eps.truncations.pad_tensor,
            timesteps=eps.timesteps.pad_tensor,
            mask=eps.mask.pad_tensor,
            id=eps.id,
            seed=eps.seed,
            env_name=eps.env_name,
            batch_size=[len(eps)],
        )

        if self.device:
            batch = batch.to(self.device)
        return batch


# helper functions for padding and stacking.  Note:
def from_eps_with_pad(eps, attr: str, batch_first: bool = True) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence([torch.from_numpy(getattr(x, attr)) for x in eps], batch_first=batch_first)


def from_eps(eps: List[EpisodeData], attr: str) -> torch.Tensor:
    return torch.tensor([getattr(x, attr) for x in eps])


def return_fn_from_episodes(batch_first: bool = True, return_class: type = Batch):
    def return_fn(eps: List[EpisodeData]):
        return return_class(
            id=from_eps(eps, "id"),
            total_timesteps=from_eps(eps, "total_timesteps"),
            states=from_eps_with_pad(eps, "states", batch_first=batch_first),
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

    return return_fn
