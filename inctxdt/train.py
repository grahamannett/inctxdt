from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import minari
import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env
from model import DecisionTransformer
from torch.nn import functional as F  # noqa
from torch.utils.data import DataLoader, Dataset

from episode_data import EpisodeData
from evaluation import eval_rollout


class config:
    device: str = "cpu"
    epochs: int = 1

    # optim
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    warmup_steps: int = 100

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


class MinariWrapper(Dataset):
    def __init__(self, env_name: str, seq_len: int = None):
        minari.download_dataset(env_name)
        self.ds = minari.load_dataset(env_name)
        self.seq_len = seq_len

    def __len__(self):
        return self.ds.total_episodes

    def __getitem__(self, idx: int) -> EpisodeData:
        episode_data = self.ds._data.get_episodes([self.ds.episode_indices[idx]])[0]

        if self.seq_len:
            episode_data = self._fix_episode(episode_data)

        if episode_data["seed"] == "None":
            episode_data["seed"] = None

        episode_data["returns_to_go"] = discounted_cumsum(episode_data["rewards"], gamma=1.0)
        episode_data["mask"] = np.ones(episode_data["total_timesteps"])
        episode_data["observations"] = episode_data["observations"][:-1]

        return EpisodeData(**episode_data)

    def get_env(self):
        return self.ds.recover_environment()


class TensorsDataclass:
    def to(self, device: str):
        return self.__class__(
            **{k: v.to(device) for k, v in self.__dict__.items() if v is not None},
        )

    def asdict(self) -> Dict[str, torch.Tensor]:
        return asdict(self)


@dataclass
class Batch(TensorsDataclass):
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

    @classmethod
    def collate_fn(cls, episodes: List[EpisodeData], device: str = None) -> "Batch":
        device = device or config.device
        return cls(
            id=torch.tensor([x.id for x in episodes], device=device),
            seed=torch.tensor([x.seed for x in episodes], device=device) if episodes[0].seed else None,
            total_timesteps=torch.tensor([x.total_timesteps for x in episodes], device=device),
            observations=torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.observations, dtype=torch.float32) for x in episodes],
                batch_first=True,
            ).to(device),
            actions=torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.actions, dtype=torch.float32) for x in episodes],
                batch_first=True,
            ).to(device),
            rewards=torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.rewards, dtype=torch.float32) for x in episodes],
                batch_first=True,
            ).to(device),
            returns_to_go=torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.returns_to_go, dtype=torch.float32) for x in episodes],
                batch_first=True,
            ).to(device),
            terminations=torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.terminations, dtype=torch.float32) for x in episodes],
                batch_first=True,
            ).to(device),
            truncations=torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.truncations, dtype=torch.float32) for x in episodes],
                batch_first=True,
            ).to(device),
            timesteps=torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.timesteps, dtype=torch.int) for x in episodes],
                batch_first=True,
            ).to(device),
            mask=torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.mask, dtype=torch.int) for x in episodes],
                batch_first=True,
            ).to(device),
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
            batch = batch.to(config.device)

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

        eval_info = eval_rollout(model, env=env, target_return=1000.0)
        print(f"Eval: {eval_info}")


if __name__ == "__main__":
    config.get()
    ds = MinariWrapper(env_name="pen-human-v0")
    dataloader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=Batch.collate_fn)
    env = ds.get_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = DecisionTransformer(state_dim=state_dim, action_dim=action_dim, embedding_dim=32, num_layers=2)
    train(model, dataloader)
