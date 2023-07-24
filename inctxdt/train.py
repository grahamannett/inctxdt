from typing import Tuple

import minari
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

from inctxdt.evaluation import eval_rollout
from inctxdt.model import DecisionTransformer
from inctxdt.batch import Batch


class config:
    device: str = "cpu"
    epochs: int = 1

    # optim
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    warmup_steps: int = 10
    batch_size: int = 4

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


def loss_fn(logits, targets, **kwargs):
    return nn.functional.mse_loss(logits, targets, **kwargs)


def train(model: nn.Module, dataloader: torch.utils.data.DataLoader):
    model = model.to(config.device)
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
        epoch_loss = 0
        for batch_idx, batch in enumerate(dataloader):
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

            epoch_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"batch-idx:{batch_idx}/{len(dataloader)} | epoch-loss: {epoch_loss:.4f}")

        eval_info = eval_rollout(model, env=env, target_return=1000.0, device=config.device)
        print(f"Eval: {eval_info}")


if __name__ == "__main__":
    config.get()
    # ds = MinariDataset(env_name="pen-human-v0")
    # env_name = "d4rl_hopper-expert-v2"
    # env_name = "d4rl_halfcheetah-expert-v2"  # test with this to make sure its working
    env_name = "pointmaze-umaze-v0"
    env_name = "pointmaze-open-dense-v0"

    # unless i update
    datasets = [
        "pointmaze-large-dense-v0",
        "pointmaze-large-v0",
        "pointmaze-medium-dense-v0",
        "pointmaze-medium-v0",
        "pointmaze-open-dense-v0",
        "pointmaze-umaze-v0",
    ]

    ds = MinariDataset(env_name=env_name)

    dataloader = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=Batch.make_collate_fn(device=config.device, batch_first=True),
    )
    env = ds.recover_environment()

    sample = ds[0]
    state_dim = sample.observations.shape[-1]
    action_dim = sample.actions.shape[-1]

    model = DecisionTransformer(state_dim=state_dim, action_dim=action_dim, embedding_dim=128, num_layers=6)
    train(model, dataloader)
