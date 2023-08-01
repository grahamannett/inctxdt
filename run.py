from torch.utils.data import DataLoader

from inctxdt.batch import Collate
from inctxdt.datasets import AcrossEpisodeDataset, MultipleMinariDataset, MinariDataset
from inctxdt.d4rl_datasets import D4rlDataset
from inctxdt.model import DecisionTransformer
from inctxdt.trainer import train
from inctxdt.config import config_tool


def main():
    config = config_tool.get()

    accelerator = None

    if config.adist:
        from accelerate import Accelerator

        accelerator = Accelerator()

    # ds = [
    #     AcrossEpisodeDataset(env_name="pointmaze-umaze-v1", max_num_epsisodes=3),
    #     AcrossEpisodeDataset(env_name="pointmaze-medium-v1", max_num_epsisodes=3),
    # ]

    # ds = MultipleMinariDataset(datasets=ds)
    # ds = D4rlDataset(env_name="halfcheetah-medium-v2")
    ds = D4rlDataset(env_name="antmaze-umaze-v2")
    # ds = MinariDataset(env_name="pointmaze-medium-v1")

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

    model = DecisionTransformer(
        state_dim=state_dim, action_dim=action_dim, embedding_dim=128, num_layers=6
    )

    train(model, dataloader=dataloader, config=config, accelerator=accelerator)


if __name__ == "__main__":
    main()
