from torch.utils.data import DataLoader


from inctxdt.batch import Collate

from inctxdt.datasets import AcrossEpisodeDataset, MultipleMinariDataset, MinariDataset
from inctxdt.d4rl_datasets import D4rlDataset
from inctxdt.model import DecisionTransformer
from inctxdt.trainer import train
from inctxdt.config import config_tool
from accelerate import Accelerator


def run_baseline(config, dataset, env_spec=None, accelerator=None):
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
    )

    train(
        model,
        dataloader=dataloader,
        config=config,
        accelerator=accelerator,
    )


def run_autoregressive(config, dataset, env_spec=None, accelerator=None):
    collater = Collate(batch_first=True, device=None if accelerator else config.device)

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
        env_spec=env_spec,
    )
    train(model, dataloader=dataloader, config=config, accelerator=accelerator)


def main():
    dataset_name = "halfcheetah-medium-v2"
    # dataset_name = "pointmaze-medium-v1"
    # ds = MinariDataset(dataset_name=dataset_name)
    config = config_tool.get(dataset_name=dataset_name)
    ds = D4rlDataset(dataset_name=dataset_name, seq_len=config.seq_len)
    print(config)

    accelerator = Accelerator()

    env_spec = config.get_env_spec()
    # sample = ds[0]

    # state_dim = sample.states.shape[-1]
    # action_dim = sample.actions.shape[-1]

    ds.state_dim = env_spec.state_dim
    ds.action_dim = env_spec.action_dim

    dispatch_cmd = {
        "train": run_autoregressive,
        "baseline": run_baseline,
    }

    dispatch_cmd[config.cmd](config, dataset=ds, env_spec=env_spec, accelerator=accelerator)

    # if config.cmd == "train":
    #     run_autoregressive(config, dataset=ds, env_spec=env_spec, accelerator=accelerator)
    # elif config.cmd == "baseline":
    #     run_baseline(config, dataset=ds, env_spec=env_spec, accelerator=accelerator)


if __name__ == "__main__":
    main()
