from torch.utils.data import DataLoader


from inctxdt.batch import Collate

from inctxdt.datasets import AcrossEpisodeDataset, MultipleMinariDataset, MinariDataset
from inctxdt.d4rl_datasets import D4rlDataset
from inctxdt.model import DecisionTransformer
from inctxdt.trainer import train
from inctxdt.config import config_tool


def run_baseline(config, dataset, accelerator=None):
    from inctxdt.baseline_dt import DecisionTransformer as DecisionTransformerBaseline

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=Collate(device=config.device, batch_first=True),
    )

    model = DecisionTransformerBaseline(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        embedding_dim=128,
        num_layers=6,
        use_single_action_head=True,
    )

    train(
        model,
        dataloader=dataloader,
        config=config,
        accelerator=accelerator,
    )


def main():
    config = config_tool.get()

    accelerator = None

    if config.adist:
        from accelerate import Accelerator

        accelerator = Accelerator()

    ds = D4rlDataset(env_name="halfcheetah-medium-v2")

    dataloader = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=Collate(device=config.device, batch_first=True),
    )

    sample = ds[0]
    batch = next(iter(dataloader))
    assert batch is not None

    state_dim = sample.observations.shape[-1]
    action_dim = sample.actions.shape[-1]

    ds.state_dim = state_dim
    ds.action_dim = action_dim
    # use_single_action_head = config.use_single_action_head

    # run_baseline(config, dataset=ds)

    model = DecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        embedding_dim=128,
        num_layers=6,
        use_single_action_head=True,
    )

    print("model, using single-action-head:", model)

    train(model, dataloader=dataloader, config=config, accelerator=accelerator)


if __name__ == "__main__":
    main()
