from typing import Callable, Tuple

import torch
import torch.nn as nn

from tqdm.auto import tqdm, trange

from gymnasium import Env

from inctxdt.evaluation import eval_rollout
from inctxdt.model import DecisionTransformer
from inctxdt.batch import Batch
from inctxdt.datasets import MinariDataset
from inctxdt.config import config_tool, EnvSpec


def loss_fn(logits, targets, **kwargs):
    return nn.functional.mse_loss(logits, targets, **kwargs)


def train(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: config_tool = config_tool,
    accelerator: type = None,
    env: Env = None,  # noqa
):
    if env is None:
        env = dataloader.dataset.recover_environment()

    env_spec = EnvSpec(  # S params is sequence params
        model.action_dim, model.state_dim, model.episode_len, model.seq_len
    )
    device = config.device
    _main_proc = accelerator.is_local_main_process if accelerator else True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )

    if accelerator:
        model, dataloader, optimizer, scheduler = accelerator.prepare(
            model, dataloader, optimizer, scheduler
        )
        device = accelerator.device
    else:
        model = torch.nn.DataParallel(model)
        model = model.to(config.device)

    def data_iter_fn(batch_iter) -> Tuple[int, Batch]:
        for batch_idx, batch in batch_iter:
            if isinstance(early_end := config.batch_early_termination, (int, float)):
                if batch_idx >= early_end:
                    return

            yield batch_idx, batch

    def handle_batch(batch) -> Batch:
        if accelerator is None:
            batch = batch.to(device)
        return batch

    def handle_loss(pred, batch) -> torch.Tensor:
        loss = loss_fn(pred, batch.actions, reduction="none")
        loss = (loss * batch.mask.unsqueeze(-1)).mean()

        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
        return loss

    def post_handle_loss(m):
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(m.parameters(), config.clip_grad)

        optimizer.step()
        scheduler.step()

    epoch_pbar = trange(
        config.epochs, desc=f"Epoch", disable=not _main_proc, leave=False
    )

    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0

        pbar = tqdm(dataloader, disable=not _main_proc)

        for batch_idx, batch in data_iter_fn(enumerate(pbar)):
            batch = handle_batch(batch)

            predicted_actions = model.forward(
                states=batch.observations,
                actions=batch.actions,
                returns_to_go=batch.returns_to_go,
                time_steps=batch.timesteps,
                mask=batch.mask,
            )

            loss = handle_loss(predicted_actions, batch)
            post_handle_loss(model)

            epoch_loss += loss.item()
            if batch_idx % 100 == 0:
                pbar.set_description(f"loss={loss.item():.4f}")

        epoch_pbar.set_description(f"Epoch{epoch} loss: {epoch_loss:.4f}")

        eval_info = eval_rollout(
            model,
            env=env,
            env_spec=env_spec,
            target_return=1000.0,
            device=device,
        )
        print(f"Eval: {eval_info}")
