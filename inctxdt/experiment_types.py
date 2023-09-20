import numpy as np
import torch
from sklearn.preprocessing import KBinsDiscretizer
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


def create_enc(arr: np.ndarray, num_bins: int = 1000, strategy: str = "quantile") -> torch.Tensor:
    enc = KBinsDiscretizer(n_bins=num_bins, encode="ordinal", strategy=strategy)
    enc.fit(arr)
    return enc


def create_bin_edges(arr: np.ndarray, num_bins: int = 1000, strategy: str = "quantile") -> torch.Tensor:
    enc = KBinsDiscretizer(n_bins=num_bins, encode="ordinal", strategy=strategy)
    enc.fit(arr)
    # bin_edges might be uneven shaped b/c of quantile strategy
    return [torch.from_numpy(b) for b in enc.bin_edges_]


def run_autoregressive(config, dataset=None, dataloader=None, accelerator=None, env_spec=None, env=None, venv=None):
    assert dataset or dataloader, "either dataset or dataloader must be provided"

    dataloader = dataloader_from_dataset(dataset, dataloader, config, accelerator=accelerator)

    # need to create discretization before creating model since we needs the vocab size which is probably action-dim*num_bins
    # enc = create_enc(
    #     arr=np.concatenate([v["actions"] for v in dataset.dataset]),
    #     num_bins=config.modal_embed.num_bins,
    #     strategy=config.modal_embed.strategy,
    # )

    # config.modal_embed.token_size = config.modal_embed.num_bins * len(enc.bin_edges_)

    bin_edges = create_bin_edges(
        arr=np.concatenate([v["actions"] for v in dataset.dataset]),
        num_bins=config.modal_embed.num_bins,
        strategy=config.modal_embed.strategy,
    )
    config.modal_embed.token_size = config.modal_embed.num_bins * len(bin_edges)

    model = DecisionTransformer(
        state_dim=env_spec.state_dim,
        action_dim=env_spec.action_dim,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        env_spec=env_spec,
        modal_embed=config.modal_embed,
    )

    model.set_discretizer("actions", bin_edges=bin_edges, device=accelerator.device)

    train(model, dataloader=dataloader, config=config, accelerator=accelerator, env_spec=env_spec, env=env, venv=venv)
