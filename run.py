import pyrallis

from accelerate import Accelerator
import torch

from torch.utils.data import DataLoader

from inctxdt.batch import Collate
from inctxdt.config import Config, EnvSpec
from inctxdt.d4rl_datasets import D4rlAcrossEpisodeDataset, D4rlDataset, D4rlMultipleDataset
from inctxdt.datasets import AcrossEpisodeDataset, MinariDataset, MultipleMinariDataset
from inctxdt.env_helper import get_env
from inctxdt.models.model import DecisionTransformer
from inctxdt.trainer import train
import wandb


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
                "settings": wandb.Settings(code_dir="inctxdt/"),  # to save models
            }
        },
    )


def run_baseline(config, dataset, accelerator=None, env_spec=None, env=None, venv=None):
    from inctxdt.baseline_dt import DecisionTransformer as DecisionTransformerBaseline

    collater = Collate(
        batch_first=True,
        device=None if accelerator else config.device,
        return_fn=dataset.collate_fn(batch_first=True),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        collate_fn=collater,
    )

    model = DecisionTransformerBaseline(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
    )

    train(model, dataloader=dataloader, config=config, accelerator=accelerator, env_spec=env_spec, env=env, venv=venv)


def run_autoregressive(config, dataset, accelerator=None, env_spec=None, env=None, venv=None):
    collater = Collate(batch_first=True, device=None if accelerator else config.device)
    collater.return_fn = dataset.collate_fn(batch_first=True)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        collate_fn=collater,
    )

    model = DecisionTransformer(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        env_spec=env_spec,
    )
    train(model, dataloader=dataloader, config=config, accelerator=accelerator, env_spec=env_spec, env=env, venv=venv)


dispatch_cmd = {
    "train": run_autoregressive,
    "baseline": run_baseline,
}

dispatch_dataset = {
    "minari": MinariDataset,
    "minari_across": AcrossEpisodeDataset,
    "minari_multiple": MultipleMinariDataset,
    "d4rl": D4rlDataset,
    "d4rl_across": D4rlAcrossEpisodeDataset,
    "d4rl_multiple": D4rlMultipleDataset,
}


def main():
    config = pyrallis.parse(config_class=Config)
    torch.cuda.manual_seed(config.seed)
    torch.manual_seed(config.seed)

    DatasetType = dispatch_dataset[config.dataset_type]
    dataset = DatasetType(dataset_name=config.dataset_name, seq_len=config.seq_len, reward_scale=config.reward_scale)

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

    accelerator = Accelerator(log_with="wandb")
    init_trackers(accelerator, config)

    accelerator.print(config)
    dispatch_cmd[config.cmd](config, dataset=dataset, accelerator=accelerator, env_spec=env_spec, env=env, venv=venv)


if __name__ == "__main__":
    main()
