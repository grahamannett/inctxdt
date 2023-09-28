from os import makedirs
from typing import Tuple, Union

# import d4rl
# import gym
import gymnasium

# NOTE: THIS IS BECAUSE IM USING GYM.MAKE HERE.  NEED TO FIX
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from inctxdt.batch import Batch
from inctxdt.config import Config, EnvSpec
from inctxdt.datasets_meta import IterDataset
from inctxdt.evaluation import venv_eval_rollout
from inctxdt.models.model_output import ModelOutput
from inctxdt.score_helpers import BestScore, EvalScore, EvalScores


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


def _secondary_loss(preds, targets, mask, scale_factor: float = 1.0):
    loss = nn.functional.mse_loss(preds, targets.detach(), reduction="none")
    if mask is not None:
        loss = loss * mask.unsqueeze(-1)

    loss = (loss * scale_factor).mean()
    return loss


def get_loss(model_output: ModelOutput, batch: Batch, config: Config = None) -> torch.Tensor:
    actions, mask, pred = batch.actions, batch.mask, model_output.logits

    if pred.shape != actions.shape:
        pred = pred.squeeze()

    # # commented out atm because loss not similar to baseline
    # if pred.shape != actions.shape:
    #     if pred.shape[-1] != actions.shape[-1]:
    #         pred = pred[..., : actions.shape[-1]]

    loss = nn.functional.mse_loss(pred, actions.detach(), reduction="none")
    loss = (loss * mask.unsqueeze(-1)).mean()

    if config.use_secondary_loss:
        if state_scale := config.state_loss_scale:
            loss += (
                nn.functional.mse_loss(model_output.extra["obs_logits"], batch.states, reduction=config.loss_reduction)
                * mask.unsqueeze(-1)
                * state_scale
            ).mean()

        if rewards_scale := config.rewards_loss_scale:
            loss += (
                nn.functional.mse_loss(model_output.extra["rewards"], batch.rewards, reduction=config.loss_reduction)
                * mask.squeeze(-1)
                * rewards_scale
            ).mean()

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

    if (optimizer is None) or (scheduler is None):
        optimizer, scheduler = default_optimizer(model, config)

    if _main_proc and config.save_model:
        makedirs(config.exp_dir, exist_ok=True)
        torch.save(model, f"{config.exp_dir}/model_base")

    # need this before we wrap as wrap seems to remove batch_size and other vars
    trainloader = DataLoader(
        dataset=IterDataset(dataset=dataloader.dataset),
        batch_size=dataloader.batch_size,
        pin_memory=dataloader.pin_memory,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
    )

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

    def post_get_loss(loss):
        optimizer.zero_grad()
        accelerator.backward(loss)
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)

        optimizer.step()
        scheduler.step()

    def _log(values: dict = None, step: int | None = None, log_kwargs: dict | None = {}):
        accelerator.log(values, step=step, log_kwargs=log_kwargs)

    def _save_model(step: Union[int, str], infos: dict = None):
        if config.save_model:
            accelerator.wait_for_everyone()
            accelerator.save_model(model, f"{config.exp_dir}/model_{step}")

            if infos:
                accelerator.save(infos, f"{config.exp_dir}/infos.pt")

    def _make_log_eval_vals(eval_ret: torch.Tensor):
        eval_ret /= config.reward_scale
        eval_score, eval_score_std = eval_ret.mean().item(), eval_ret.std().item()
        norm_score, norm_score_std = None, None

        if hasattr(env, "get_normalized_score"):
            norm_score = env.get_normalized_score(eval_ret) * 100
            norm_score, norm_score_std = norm_score.mean().item(), norm_score.std().item()

        return eval_score, eval_score_std, norm_score, norm_score_std

    def eval_fn(step) -> dict:
        scores = EvalScores()

        for target_return in config.target_returns:
            eval_ret, _ = venv_eval_rollout(
                model,
                venv,
                env_spec,
                target_return=target_return * config.reward_scale,
                device=accelerator.device,
                output_sequential=config.eval_output_sequential,
                seed=config.eval_seed,
            )

            eval_score, eval_score_std, norm_score, norm_score_std = _make_log_eval_vals(eval_ret)
            _log(
                {
                    f"eval/{target_return}_return_mean": eval_score,
                    f"eval/{target_return}_return_std": eval_score_std,
                    f"eval/{target_return}_normalized_score_mean": norm_score,
                    f"eval/{target_return}_normalized_score_std": norm_score_std,
                },
                step=step,
            )
            scores.target[f"{target_return}"] = EvalScore(eval_score, eval_score_std, norm_score, norm_score_std)

        model.train()
        return scores

    train_iter = iter(trainloader)
    _save_model("init")

    pbar = trange(config.update_steps, desc=f"Training", disable=not _main_proc, leave=False)
    best_score = BestScore()
    infos = {}
    for step in pbar:
        model.train()
        batch: Batch = next(train_iter)
        batch = batch.to(accelerator.device)
        model_output = model(
            states=batch.states,
            actions=batch.actions,
            returns_to_go=batch.returns_to_go,
            timesteps=batch.timesteps,
            padding_mask=batch.make_padding_mask(),
        )

        loss = get_loss(model_output, batch, config=config)
        post_get_loss(loss)
        last_lr = scheduler.get_last_lr()[0]

        # eval
        if (step % config.eval_every) == 0:
            scores: EvalScores = eval_fn(step)
            eval_score, norm_score = scores.mean_eval_score, scores.mean_normalized_score
            infos[f"eval/{step}"] = scores

            best_score.update(scores=scores, step=step)

        if (step % config.log.log_every) == 0:
            _log({"loss": loss.item(), "learning_rate": last_lr}, step=step)

        _save_model(step, infos=infos)

        pbar.set_postfix_str(
            f"[S:{step}][L:{loss.item():.4f}]|->eval:{eval_score:.2f}|->norm:{norm_score:.2f}|->lr:{last_lr:.4f}"
        )
        # accelerator.print(f"[S:{step}][L:{loss.item():.4f}]|->eval:{eval_score:.2f}|->norm:{norm_score:.2f}")

    accelerator.print(f"best scores: {best_score}")
