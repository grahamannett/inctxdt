from torch.utils.data import DataLoader


from inctxdt.batch import Collate

from inctxdt.datasets import AcrossEpisodeDataset, MultipleMinariDataset, MinariDataset
from inctxdt.d4rl_datasets import D4rlDataset
from inctxdt.model import DecisionTransformer
from inctxdt.trainer import train
from inctxdt.config import config_tool
from accelerate import Accelerator


def run_baseline(config, dataset, accelerator=None):
    from inctxdt.baseline_dt import DecisionTransformer as DecisionTransformerBaseline

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=Collate(device=config.device, batch_first=True),
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


def run_autoregressive(config, dataset, accelerator=None):
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=Collate(device=config.device, batch_first=True),
    )

    model = DecisionTransformer(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        embedding_dim=128,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
    )
    train(model, dataloader=dataloader, config=config, accelerator=accelerator)


def main():
    config = config_tool.get()

    accelerator = Accelerator()

    ds = D4rlDataset(env_name="halfcheetah-medium-v2")
    # ds = D4rlDataset(env_name="pointmaze-medium-v1")
    sample = ds[0]

    state_dim = sample.observations.shape[-1]
    action_dim = sample.actions.shape[-1]

    ds.state_dim = state_dim
    ds.action_dim = action_dim

    if config.cmd == "train":
        run_autoregressive(config, dataset=ds, accelerator=accelerator)
    elif config.cmd == "baseline":
        run_baseline(config, dataset=ds, accelerator=accelerator)


if __name__ == "__main__":
    main()
