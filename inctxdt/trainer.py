from typing import Optional, Tuple, Union

from os import makedirs

# import d4rl
# import gym
import gymnasium
from accelerate import Accelerator
import gc


# NOTE: THIS IS BECAUSE IM USING GYM.MAKE HERE.  NEED TO FIX
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from inctxdt.batch import Batch
from inctxdt.config import EnvSpec, Config
from inctxdt.evaluation import venv_eval_rollout
from inctxdt.models.model_output import ModelOutput
from fast_pytorch_kmeans import KMeans


def default_optimizer(model, config):
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
    return optimizer, scheduler


def _secondary_loss(
    obs_pred: torch.Tensor,
    obs_target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale_factor: float = 1.0,
):
    loss = nn.functional.mse_loss(obs_pred, obs_target, reduction="none")
    if mask is not None:
        loss = (loss * mask.unsqueeze(-1)).mean()
    loss *= scale_factor
    return loss


def get_loss(model_output: ModelOutput, batch: Batch, config: Config = None) -> torch.Tensor:
    target, mask, pred = batch.actions, batch.mask, model_output.logits

    if pred.shape != target.shape:
        pred = pred.squeeze()

    if pred.shape != target.shape:
        breakpoint()

    loss = nn.functional.mse_loss(pred, target, reduction="none")
    loss = (loss * mask.unsqueeze(-1)).mean()

    if getattr(config, "use_secondary_loss", False):
        loss += _secondary_loss(
            model_output.extra["obs_logits"],
            batch.states,
            mask,
            scale_factor=getattr(config, "secondary_loss_scale", 1.0),
        )

    return loss


def train(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: Config,
    accelerator: Accelerator = None,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR = None,
    env_spec: EnvSpec = None,
    env: gymnasium.Env = None,
    venv: gymnasium.vector.SyncVectorEnv = None,
):
    _main_proc = accelerator.is_local_main_process
    eval_rollout_fn = getattr(config, "eval_func", venv_eval_rollout)

    if (optimizer is None) or (scheduler is None):
        optimizer, scheduler = default_optimizer(model, config)

    if _main_proc and config.save_model:
        makedirs(config.exp_dir, exist_ok=True)
        torch.save(model, f"{config.exp_dir}/model_base")

    model, dataloader, optimizer, scheduler = accelerator.prepare(
        model,
        dataloader,
        optimizer,
        scheduler,
    )

    # functions that use accelerator/model/etc objects that are not passed in as args
    def data_iter_fn(batch_iter) -> Tuple[int, Batch]:
        for batch_idx, batch in batch_iter:
            if (config.n_batches not in [-1, None]) and (batch_idx > config.n_batches):
                return

            yield batch_idx, batch

    def post_get_loss(mod, loss):
        optimizer.zero_grad()
        accelerator.backward(loss)
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(mod.parameters(), config.clip_grad)

        optimizer.step()
        scheduler.step()

    def _log(**kwargs):
        accelerator.log(kwargs)

    def _norm_score(eval_returns: torch.Tensor) -> float:
        if hasattr(env, "get_normalized_score"):
            return (env.get_normalized_score(eval_returns) * 100).mean().item()
        return 0

    def _save_model(epoch: Union[int, str]):
        if config.save_model:
            accelerator.wait_for_everyone()
            accelerator.save_model(model, f"{config.exp_dir}/model_{epoch}")

    # epoch_pbar = trange(config.epochs, desc=f"Epoch", disable=not _main_proc, leave=False)
    epoch_pbar = range(config.epochs)

    _save_model("init")

    for epoch in epoch_pbar:
        epoch_loss = 0

        pbar = tqdm(dataloader, disable=not _main_proc)

        model.train()
        batch: Batch
        for batch_idx, batch in data_iter_fn(enumerate(pbar)):
            model_output = model.forward(
                states=batch.states,
                actions=batch.actions,
                returns_to_go=batch.returns_to_go,
                timesteps=batch.timesteps,
                padding_mask=batch.make_padding_mask(),
            )

            loss = get_loss(model_output, batch)
            post_get_loss(model, loss)

            epoch_loss += loss.item()
            if batch_idx % 100 == 0:
                pbar.set_description(f"loss={loss.item():.4f}")

            if (batch_idx % config.log.log_every) == 0:
                _log(loss=loss.item())

        eval_ret, _ = eval_rollout_fn(
            model,
            venv,
            env_spec,
            target_return=config.target_return * config.reward_scale,
            device=accelerator.device,
            output_sequential=config.eval_output_sequential,
        )

        eval_ret /= config.reward_scale
        eval_ret_mean = eval_ret.mean().item()
        norm_score = _norm_score(eval_ret)

        _log(epoch=epoch, returns=eval_ret_mean, norm_score=norm_score if norm_score != 0 else None)

        accelerator.print(f"[E:{epoch}][L:{epoch_loss:.4f}]|->eval:{eval_ret_mean:.2f}|->norm:{norm_score:.2f}")

        _save_model(epoch)
