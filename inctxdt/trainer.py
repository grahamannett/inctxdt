from typing import Tuple, Union

# import d4rl
# import gym
import gymnasium

import time

# NOTE: THIS IS BECAUSE IM USING GYM.MAKE HERE.  NEED TO FIX
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange

from inctxdt.batch import Batch
from inctxdt.config import EnvSpec, config_tool
from inctxdt.evaluation import eval_rollout, venv_eval_rollout
from inctxdt.model_output import ModelOutput

from inctxdt.env_helper import get_env


def train(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: config_tool = config_tool,
    accelerator: type = None,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR = None,
    env_spec: EnvSpec = None,
    env: gymnasium.Env = None,
    venv: gymnasium.vector.SyncVectorEnv = None,
):
    if env is None or env_spec is None:
        _env_fn, env, venv, obs_dim, act_dim = get_env(dataset=dataloader.dataset, config=config)

        env_spec = EnvSpec(
            action_dim=act_dim.shape[0],  #  or model.action_dim
            state_dim=obs_dim.shape[0],  # or model.state_dim
            env_name=dataloader.dataset.dataset_name,
            episode_len=model.episode_len,
            seq_len=model.seq_len,
        )

    _main_proc = accelerator.is_local_main_process

    if optimizer is None:
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

    model, dataloader, optimizer, scheduler = accelerator.prepare(model, dataloader, optimizer, scheduler)

    def data_iter_fn(batch_iter) -> Tuple[int, Batch]:
        for batch_idx, batch in batch_iter:
            if (config.n_batches not in [-1, None]) and (batch_idx > config.n_batches):
                return

            yield batch_idx, batch

    def handle_loss(model_output: ModelOutput, batch: Batch) -> torch.Tensor:
        target, mask, pred = batch.actions, batch.mask, model_output.logits

        # for mse we need shapes equal
        if pred.shape != target.shape:
            pred = pred.reshape(target.shape)

        loss = nn.functional.mse_loss(pred, target, reduction="none")
        loss = (loss * mask.unsqueeze(-1)).mean()

        accelerator.backward(loss)
        return loss

    def post_handle_loss(m):
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(m.parameters(), config.clip_grad)

        optimizer.step()
        scheduler.step()

    epoch_pbar = trange(config.epochs, desc=f"Epoch", disable=not _main_proc, leave=False)

    for epoch in epoch_pbar:
        epoch_loss = 0

        pbar = tqdm(dataloader, disable=not _main_proc)

        model.train()
        batch: Batch
        for batch_idx, batch in data_iter_fn(enumerate(pbar)):
            # padding_mask = ~batch.mask.to(torch.bool)

            model_output = model.forward(
                states=batch.states,
                actions=batch.actions,
                returns_to_go=batch.returns_to_go,
                timesteps=batch.timesteps,
                padding_mask=batch.make_padding_mask(),
            )

            loss = handle_loss(model_output, batch)
            post_handle_loss(model)

            epoch_loss += loss.item()
            if batch_idx % 100 == 0:
                pbar.set_description(f"loss={loss.item():.4f}")

        epoch_pbar.set_description(f"Epoch{epoch} loss: {epoch_loss:.4f}")

        eval_ret, eval_len = venv_eval_rollout(model, venv, env_spec, target_return=12000.0, device=accelerator.device)
        eval_ret = eval_ret / config.reward_scale

        norm_score = 0.0

        if hasattr(env, "get_normalized_score"):
            norm_score = (env.get_normalized_score(eval_ret) * 100).mean().item()

        accelerator.print(
            f"\n==>Epoch{epoch} loss: {epoch_loss:.4f} | Eval: {eval_ret.mean().item():.2f} len: {eval_len.float().mean().item():.2f} norm-score {norm_score:.2f}"
        )
        epoch_pbar.set_description(
            f"Epoch{epoch} loss: {epoch_loss:.4f} eval: {eval_ret.mean().item():.2f} | norm_score {norm_score:.2f}"
        )
