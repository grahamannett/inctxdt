from accelerate import Accelerator
from torch.utils.data import DataLoader

from inctxdt.batch import Collate
from inctxdt.config import EnvSpec, config_tool
from inctxdt.d4rl_datasets import D4rlAcrossEpisodeDataset, D4rlDataset
from inctxdt.datasets import AcrossEpisodeDataset, MinariDataset, MultipleMinariDataset
from inctxdt.env_helper import get_env
from inctxdt.model import DecisionTransformer
from inctxdt.trainer import train


def run_baseline(config, dataset, accelerator=None, env_spec=None, env=None, venv=None):
    from inctxdt.baseline_dt import DecisionTransformer as DecisionTransformerBaseline

    collater = Collate(batch_first=True, device=None if accelerator else config.device)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collater,
    )

    model = DecisionTransformerBaseline(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        embedding_dim=128,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        seq_len=config.seq_len,
    )

    train(model, dataloader=dataloader, config=config, accelerator=accelerator, env_spec=env_spec, env=env, venv=venv)


def run_autoregressive(config, dataset, accelerator=None, env_spec=None, env=None, venv=None):
    collater = Collate(batch_first=True, device=None if accelerator else config.device)
    collater.return_fn = dataset.make_batch_return_fn(batch_first=True)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collater,
    )

    model = DecisionTransformer(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        embedding_dim=128,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        seq_len=config.seq_len,
        env_spec=env_spec,
    )
    train(model, dataloader=dataloader, config=config, accelerator=accelerator, env_spec=env_spec, env=env, venv=venv)


def main():
    # dataset_name = "pointmaze-medium-v1"
    dataset_name = "halfcheetah-medium-v2"
    config = config_tool.get(dataset_name=dataset_name)
    # ds = MinariDataset(dataset_name=config.dataset_name)
    ds = D4rlAcrossEpisodeDataset(dataset_name=dataset_name, seq_len=config.seq_len)

    print(config)

    accelerator = Accelerator()

    env_fn, env, venv, obs_space, act_space = get_env(dataset=ds, config=config)

    env_spec = EnvSpec(
        episode_len=config.episode_len,
        seq_len=config.seq_len,
        env_name=getattr(ds, "env_name", ds.dataset_name),
        action_dim=act_space.shape[0],
        state_dim=obs_space.shape[0],
    )

    ds.state_dim = env_spec.state_dim
    ds.action_dim = env_spec.action_dim

    dispatch_cmd = {
        "train": run_autoregressive,
        "baseline": run_baseline,
    }

    dispatch_cmd[config.cmd](config, dataset=ds, accelerator=accelerator, env_spec=env_spec, env=env, venv=venv)


if __name__ == "__main__":
    main()
