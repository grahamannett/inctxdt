import random
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import minari
import numpy as np
import torch
import os
import torch.nn as nn
from gymnasium import Env
from minari.storage.datasets_root_dir import get_dataset_path
from torch.nn import functional as F  # noqa
from torch.utils.data import DataLoader, Dataset

from inctxdt.episode_data import EpisodeData
from inctxdt.evaluation import eval_rollout
from inctxdt.model import DecisionTransformer


class config:
    device: str = "cpu"
    epochs: int = 1

    # optim
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    warmup_steps: int = 10

    clip_grad: bool = True

    @classmethod
    def get(cls):
        import argparse

        parser = argparse.ArgumentParser()
        for k, v in cls.__dict__.items():
            if k.startswith("_"):
                continue
            parser.add_argument(f"--{k}", type=type(v), default=v)
        args = parser.parse_args()

        for k, v in vars(args).items():
            setattr(cls, k, v)
        return cls


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


class MinariDataset(minari.MinariDataset):
    def __init__(self, env_name: str, seq_len: int = None):
        data_path = self._get_data_path_from_env_name(env_name)
        super().__init__(data=data_path)

        self.env_name = env_name
        self.seq_len = seq_len

    def __len__(self):
        return self.total_episodes

    def __getitem__(self, idx: int, output_cls: Callable = EpisodeData) -> EpisodeData:
        (episode_data,) = self._data.get_episodes([self.episode_indices[idx]])
        episode_data = self._fix_episode(episode_data)
        return output_cls(**episode_data)

    def _get_data_path_from_env_name(self, env_name: str) -> str:
        if env_name not in minari.list_local_datasets():
            minari.download_dataset(env_name)
        file_path = get_dataset_path(env_name)
        data_path = os.path.join(file_path, "data", "main_data.hdf5")
        return data_path

    def _fix_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.seq_len:
            raise NotImplementedError

        if episode_data["seed"] == "None":
            episode_data["seed"] = None

        episode_data["returns_to_go"] = discounted_cumsum(episode_data["rewards"], gamma=1.0)
        episode_data["mask"] = np.ones(episode_data["total_timesteps"])
        episode_data["observations"] = episode_data["observations"][:-1]
        episode_data["env_name"] = self.env_name
        return episode_data


class AcrossEpisodeDataset(MinariDataset):
    def __init__(self, env_name: str, seq_len: int = None, max_num_epsisodes: int = 3, drop_last: bool = False):
        super().__init__(env_name, seq_len)
        self.max_num_episodes = max_num_epsisodes

        self.drop_last = drop_last

    def __getitem__(self, idx: int) -> Any:
        ep_idxs = [self.ds.episode_indices[idx : idx + self.max_num_episodes]]
        episode_data = self.ds._data.get_episodes(*ep_idxs)

        to_get = self.max_num_episodes
        episodes = [super().__getitem__(idx, output_cls=dict)]

        while to_get > 0:
            if (idx := idx + 1) >= len(self):
                if self.drop_last:
                    break
                idx = 0

            episode = super().__getitem__(idx, output_cls=dict)
            episodes.append(episode)
            to_get -= 1
        return episodes


class SamplesDataclass:
    def to(self, device: str):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)
        return self

    def asdict(self) -> Dict[str, torch.Tensor]:
        return asdict(self)


@dataclass
class Batch(SamplesDataclass):
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
    def collate_fn(cls, episodes: List[EpisodeData], device: str = None, batch_first: bool = True) -> "Batch":
        device = device or config.device

        return cls(
            id=torch.tensor([x.id for x in episodes], device=device),
            seed=torch.tensor([x.seed for x in episodes], device=device) if episodes[0].seed else None,
            total_timesteps=torch.tensor([x.total_timesteps for x in episodes], device=device),
            observations=torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.observations, dtype=torch.float32) for x in episodes],
                batch_first=batch_first,
            ).to(device),
            actions=torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.actions, dtype=torch.float32) for x in episodes],
                batch_first=batch_first,
            ).to(device),
            rewards=torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.rewards, dtype=torch.float32) for x in episodes],
                batch_first=batch_first,
            ).to(device),
            returns_to_go=torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.returns_to_go, dtype=torch.float32) for x in episodes],
                batch_first=batch_first,
            ).to(device),
            terminations=torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.terminations, dtype=torch.int) for x in episodes],
                batch_first=batch_first,
            ).to(device),
            truncations=torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.truncations, dtype=torch.int) for x in episodes],
                batch_first=batch_first,
            ).to(device),
            timesteps=torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.timesteps, dtype=torch.int) for x in episodes],
                batch_first=batch_first,
            ).to(device),
            mask=torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.mask, dtype=torch.int) for x in episodes],
                batch_first=batch_first,
            ).to(device),
            env_name=[x.env_name for x in episodes],
        )


def loss_fn(logits, actions, **kwargs):
    return nn.functional.mse_loss(logits, actions, **kwargs)


def train(model: nn.Module, dataloader: torch.utils.data.DataLoader):
    model.to(config.device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )

    for epoch in range(config.epochs):
        print(f"Epoch {epoch}")
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            # batch = batch.to(config.device)

            padding_mask = ~batch.mask.to(torch.bool)

            predicted_actions = model.forward(
                states=batch.observations,
                actions=batch.actions,
                returns_to_go=batch.returns_to_go,
                time_steps=batch.timesteps,
                padding_mask=padding_mask,
            )

            loss = loss_fn(predicted_actions, batch.actions, reduction="none")
            loss = (loss * batch.mask.unsqueeze(-1)).mean()
            loss.backward()

            if config.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optim.step()
            scheduler.step()

        eval_info = eval_rollout(model, env=env, target_return=1000.0, device=config.device)
        print(f"Eval: {eval_info}")


if __name__ == "__main__":
    config.get()
    # ds = MinariDataset(env_name="pen-human-v0")
    # env_name = "d4rl_hopper-expert-v2"
    env_name = "d4rl_halfcheetah-expert-v2"
    env_name = "pointmaze-umaze-v0"

    ds = MinariDataset(env_name=env_name)

    dataloader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=Batch.collate_fn)
    env = ds.recover_environment()

    sample = ds[0]
    state_dim = sample.observations.shape[-1]
    action_dim = sample.actions.shape[-1]

    model = DecisionTransformer(state_dim=state_dim, action_dim=action_dim, embedding_dim=128, num_layers=6)
    train(model, dataloader)
