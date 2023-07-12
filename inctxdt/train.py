import random
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


class EnvEmbedding(nn.Module):
    def __init__(self, env: Env, embedding_dim: int, episode_len: int = 1000):
        super().__init__()
        state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]

        self.embedding_dict = nn.ModuleDict(
            {
                "timesteps": nn.Embedding(episode_len, embedding_dim),
                "observations": nn.Linear(state_dim, embedding_dim),
                "actions": nn.Linear(action_dim, embedding_dim),
                "rewards": nn.Linear(1, embedding_dim),
            }
        )

    def make_timesteps(self, total_timesteps: int) -> torch.Tensor:
        return torch.arange(total_timesteps)

    def forward(self, samples: EpisodeData) -> torch.Tensor:
        if not hasattr(samples, "timesteps"):
            samples.timesteps = torch.arange(samples.actions.shape[0])

        embs = {
            "observations": self.embedding_dict["observations"](samples.observations),
            "actions": self.embedding_dict["actions"](samples.actions),
            "rewards": self.embedding_dict["rewards"](samples.returns_to_go.unsqueeze(-1)),
            "timesteps": self.embedding_dict["timesteps"](samples.timesteps),
        }
        observation_embs = self.embedding_dict["observations"](samples.observations)
        action_embs = self.embedding_dict["actions"](samples.actions)
        reward_embs = self.embedding_dict["rewards"](samples.returns_to_go.unsqueeze(-1))
        timestep_embs = self.embedding_dict["timesteps"](samples.timesteps)

        reward_embs += timestep_embs[:, 1:, :]
        action_embs += timestep_embs[:, 1:, :]
        observation_embs += timestep_embs

        embs = torch.stack([reward_embs, observation_embs, action_embs], dim=1)

        # embs = torch.cat(list(embs.values()), dim=1)
        # returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        # # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        # sequence = (
        #     torch.stack([returns_emb, state_emb, act_emb], dim=1)
        #     .permute(0, 2, 1, 3)
        #     .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        # )
        return embs


class Model(nn.Module):
    def __init__(
        self,
        env: Env,
        embedding_dim: int = 64,
    ):
        super().__init__()

        # these could be wrapped into one module or moduledict per env
        self.env_embedding = EnvEmbedding(env, embedding_dim=embedding_dim)
        self.head = nn.Linear(embedding_dim, env.action_space.shape[0])

        self.transformer = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=4), num_layers=3)

    def forward(self, episode):
        emb = self.env_embedding(episode)
        emb = self.transformer(emb, emb)
        emb = self.head(emb)
        return emb


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
            id=torch.Tensor([x.id for x in episodes], device=device),
            seed=torch.Tensor([x.seed for x in episodes], device=device) if episodes[0].seed else None,
            total_timesteps=torch.Tensor([x.total_timesteps for x in episodes], device=device),
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
    # pen-human-v0 think this is smaller
    # "CartPole-v1-random"
    # batch = next(iter(dataloader))
    # batch = batch.to("cpu")

    ds = MinariWrapper(env_name="pen-human-v0")
    dataloader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=Batch.collate_fn)
    env = ds.get_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = DecisionTransformer(state_dim=state_dim, action_dim=action_dim, embedding_dim=32, num_layers=2)
    train(model, dataloader)
