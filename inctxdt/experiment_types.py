from torch.utils.data import DataLoader

from inctxdt.batch import Collate
from inctxdt.models.model import DecisionTransformer
from inctxdt.trainer import train


def dataloader_from_dataset(dataset, dataloader=None, config=None, accelerator=None):
    if dataloader:
        return dataloader

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
    return dataloader


def run_baseline(config, dataset=None, dataloader=None, accelerator=None, env_spec=None, env=None, venv=None):
    assert dataset or dataloader, "either dataset or dataloader must be provided"
    from inctxdt.baseline_dt import DecisionTransformer as DecisionTransformerBaseline

    dataloader = dataloader_from_dataset(dataset, dataloader, config, accelerator=accelerator)

    model = DecisionTransformerBaseline(
        state_dim=env_spec.state_dim,
        action_dim=env_spec.action_dim,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
    )

    train(model, dataloader=dataloader, config=config, accelerator=accelerator, env_spec=env_spec, env=env, venv=venv)


def run_autoregressive(config, dataset=None, dataloader=None, accelerator=None, env_spec=None, env=None, venv=None):
    assert dataset or dataloader, "either dataset or dataloader must be provided"

    dataloader = dataloader_from_dataset(dataset, dataloader, config, accelerator=accelerator)

    model = DecisionTransformer(
        state_dim=env_spec.state_dim,
        action_dim=env_spec.action_dim,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        env_spec=env_spec,
        EmbedClass=config.embed_class,
    )

    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=config.batch_size * 4,
                shuffle=config.shuffle,
                num_workers=config.num_workers,
                collate_fn=Collate(
                    batch_first=True,
                    device=None if accelerator else config.device,
                    return_fn=dataset.collate_fn(batch_first=True),
                ),
            )
        )
    )

    model.create_discretizer("actions", batch.actions)
    train(model, dataloader=dataloader, config=config, accelerator=accelerator, env_spec=env_spec, env=env, venv=venv)
