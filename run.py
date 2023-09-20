import pyrallis
import torch
import wandb
from accelerate import Accelerator

from inctxdt.config import Config, EnvSpec
from inctxdt.d4rl_datasets import D4rlAcrossEpisodeDataset, D4rlDataset, D4rlMultipleDataset
from inctxdt.env_helper import get_env
from inctxdt.experiment_types import run_autoregressive, run_baseline
from inctxdt.minari_datasets import AcrossEpisodeDataset, MinariDataset, MultipleMinariDataset

dispatch_dataset = {
    "minari": MinariDataset,
    "minari_across": AcrossEpisodeDataset,
    "minari_multiple": MultipleMinariDataset,
    "d4rl": D4rlDataset,
    "d4rl_across": D4rlAcrossEpisodeDataset,
    "d4rl_multiple": D4rlMultipleDataset,
}

dispatch_cmd = {
    "train": run_autoregressive,
    "baseline": run_baseline,
}


def init_trackers(accelerator, config):
    if accelerator is None:
        return

    accelerator.init_trackers(
        project_name=config.log.project,
        init_kwargs={
            "wandb": {
                "name": config.log.name,
                # "id": str(uuid.uuid4()), # dont save id if you want the wandb names
                "group": config.log.group,
                "mode": config.log.mode,
                "tags": config.log.tags,
                # to save models code in wandb
                "settings": wandb.Settings(code_dir="inctxdt/"),
            }
        },
    )


def make_dataset_from_config(config: Config, dataset_name: str = None):
    dataset_name = dataset_name or config.dataset_name

    if isinstance(dataset_name, list):
        dataset = D4rlMultipleDataset([make_dataset_from_config(config, name) for name in dataset_name])
        return dataset

    DatasetType = dispatch_dataset[config.dataset_type]
    dataset = DatasetType(
        dataset_name=dataset_name,
        seq_len=config.seq_len,
        reward_scale=config.reward_scale,
        max_num_episodes=config.max_num_episodes,
        min_length=config.dataset_min_length,
    )

    return dataset


def main():
    config = pyrallis.parse(config_class=Config)
    torch.cuda.manual_seed(config.seed)
    torch.manual_seed(config.seed)

    dataset = make_dataset_from_config(config)

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

    # usually i try to init tracker last so i have last few seconds to exit script if needed
    accelerator = Accelerator(log_with="wandb")
    init_trackers(accelerator, config)

    accelerator.print(config)
    dispatch_cmd[config.cmd](config, dataset=dataset, accelerator=accelerator, env_spec=env_spec, env=env, venv=venv)


if __name__ == "__main__":
    main()
