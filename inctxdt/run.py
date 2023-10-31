import os
import random
import warnings

import numpy as np
import pyrallis
import torch
from accelerate import Accelerator

import wandb
from inctxdt.config import Config, EnvSpec
from inctxdt.d4rl_datasets import D4rlAcrossEpisodeDataset, D4rlDataset, D4rlMultipleDataset
from inctxdt.env_helper import get_env
from inctxdt.experiment_types import dataloader_from_dataset, run_autoregressive, run_baseline
from inctxdt.trainer import train, default_scheduler, default_optimizer
from inctxdt.minari_datasets import AcrossEpisodeDataset, MinariDataset, MultipleMinariDataset

warnings.filterwarnings("ignore", category=DeprecationWarning)

dataset_cls = {
    "minari": MinariDataset,
    "minari_across": AcrossEpisodeDataset,
    "minari_multiple": MultipleMinariDataset,
    "d4rl": D4rlDataset,
    "d4rl_across": D4rlAcrossEpisodeDataset,
    "d4rl_multiple": D4rlMultipleDataset,
}


# note: cant set env seed when its venv
def set_seed(seed: int, deterministic_torch: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.use_deterministic_algorithms(deterministic_torch)


def init_trackers(accelerator, config):
    if accelerator is None:
        return

    name = f"{config.log.name}-{config.log.id}" if config.log.name else None

    accelerator.init_trackers(
        project_name=config.log.project,
        init_kwargs={
            "wandb": {
                # dont save id or name if you want the wandb names
                # "id": str(uuid.uuid4()), f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
                "name": name,
                "mode": config.log.mode,
                "group": config.log.group,
                "tags": config.log.tags,
                "job_type": config.log.job_type,
                # to save models code in wandb
                "settings": wandb.Settings(code_dir="inctxdt/"),
            }
        },
    )


def make_dataset_from_config(config: Config, dataset_name: str = None):
    dataset_name = dataset_name or config.env_name

    if isinstance(dataset_name, list):
        dataset = D4rlMultipleDataset([make_dataset_from_config(config, name) for name in dataset_name])
        return dataset

    DatasetType = dataset_cls[config.dataset_type]

    dataset = DatasetType(
        dataset_name=dataset_name,
        seq_len=config.seq_len,
        reward_scale=config.reward_scale,
        max_num_episodes=config.max_num_episodes,
        min_length=config.dataset_min_length,
    )

    return dataset


def run_downstream(config, dataset=None, dataloader=None, accelerator=None, env_spec=None, env=None, venv=None):
    # run original training

    # returns: model, optimizer, scheduler, infos
    model, _, _, infos = run_autoregressive(
        config,
        dataset,
        dataloader,
        accelerator,
        env_spec,
        env,
        venv,
        # args passed into train()
        skip_log=True,
        skip_eval=True,
    )

    accelerator.print("Pre-training done with best", infos["best"])

    # setup downstream comparison
    config.setup_downstream(config.downstream)
    downstream_dataset = make_dataset_from_config(config, dataset_name=config.downstream.env_name)

    config.state_mean = downstream_dataset.state_mean
    config.state_std = downstream_dataset.state_std

    _, downstream_env, downstream_venv, downstream_obs_space, downstream_act_space = get_env(
        config=config,
        dataset=downstream_dataset,
    )

    downstream_env_spec = EnvSpec(
        episode_len=config.episode_len,
        seq_len=config.seq_len,
        env_name=getattr(downstream_dataset, "env_name", downstream_dataset.dataset_name),
        state_dim=downstream_obs_space.shape[0],
        action_dim=downstream_act_space.shape[0],
    )

    downstream_dataloader = dataloader_from_dataset(downstream_dataset, config=config, accelerator=accelerator)

    # before we patch the layer, we swap the tokenizer and remove the old path/params for states/actions
    if config.downstream.patch_states:
        model.embed_paths.branches.states = None

    if config.modal_embed.tokenize_action:
        # we likely need to generate new tokenizer as dim is different for downstream dataset
        # model.embed_paths.branches.actions = None

        # we might patch actions but there is no tokenizer e.g. for non-tokenized models
        if hasattr(model.discretizers, "new_tokenizer"):
            model.discretizers.new_tokenizer(
                "actions",
                data=np.concatenate([v["actions"] for v in downstream_dataset.dataset]),
                per_column=config.modal_embed.per_action_encode,
            )

    param_groups = []
    param_groups_names = []
    base_model_params = list(model.parameters())

    if config.downstream.patch_states:
        del model.embed_paths.branches["states"]
        states_branch = model.embed_paths.new_branch(
            "states",
            new_branch=torch.nn.Linear(in_features=downstream_obs_space.shape[0], out_features=config.embedding_dim),
        )

    if config.downstream.update_optim_states:
        param_groups.append({"params": model.embed_paths.branches.states.parameters(), "lr": config.learning_rate})
        param_groups_names.append("states")

    if config.downstream.patch_actions:
        del model.embed_paths.branches["actions"]
        actions_branch = model.embed_paths.new_branch(
            "actions",
            action_dim=downstream_act_space.shape[0],
        )

    if config.downstream.update_optim_actions:
        param_groups.append({"params": model.embed_paths.branches.actions.parameters(), "lr": config.learning_rate})
        param_groups_names.append("actions")

    # if we are only patching states/actions then we dont want to update the base model params
    if config.downstream.optim_only_patched is False:
        param_groups.append({"params": base_model_params, "lr": config.learning_rate})
        param_groups_names.append("base")

    if config.downstream.optim_use_default:
        optimizer, skipped_named = default_optimizer(model, config)
        if skipped_named:
            accelerator.print("Skipped names For Downstream:", skipped_named)
        param_groups_names = ["all"]
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

    if optimizer is not None:
        scheduler = default_scheduler(optimizer, config)

    accelerator.print("Downstream task...", downstream_env_spec)
    torch.cuda.empty_cache()
    train(
        model,
        dataloader=downstream_dataloader,
        config=config,
        accelerator=accelerator,
        env_spec=downstream_env_spec,
        env=downstream_env,
        venv=downstream_venv,
        optimizer=optimizer,
        scheduler=scheduler,
        reset_scheduler=config.downstream.reset_sched,
    )


def main():
    config = pyrallis.parse(config_class=Config)

    set_seed(config.train_seed, config.deterministic_torch)

    dataset = make_dataset_from_config(config)
    config.state_mean = dataset.state_mean
    config.state_std = dataset.state_std

    _, env, venv, obs_space, act_space = get_env(config=config, dataset=dataset)

    env_spec = EnvSpec(
        episode_len=config.episode_len,
        seq_len=config.seq_len,
        env_name=getattr(dataset, "env_name", dataset.dataset_name),
        action_dim=act_space.shape[0],
        state_dim=obs_space.shape[0],
    )

    dataset.state_dim = env_spec.state_dim
    dataset.action_dim = env_spec.action_dim

    # usually i try to init tracker last so theres a few seconds to exit script if needed
    accelerator = Accelerator(log_with="wandb")
    init_trackers(accelerator, config)

    # print config
    accelerator.print(config)

    run_fns = {
        "train": run_autoregressive,
        "baseline": run_baseline,
        "downstream": run_downstream,
    }

    run_fns[config.cmd](config, dataset=dataset, accelerator=accelerator, env_spec=env_spec, env=env, venv=venv)


if __name__ == "__main__":
    main()
