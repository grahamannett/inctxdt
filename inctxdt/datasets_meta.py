from dataclasses import asdict

import torch

from inctxdt.config import Config
from torch.utils.data import Dataset


class Discretizer:
    def create_discretizer(
        self, type_name, arr: torch.Tensor, n_bins: int = 1024, eps: float = 1e-6, range: tuple[int] = (-1, 1)
    ):
        if not hasattr(self, "discretizers"):
            self.discretizers = {}

        hist = torch.histogram(arr.view(-1), bins=n_bins, range=(range[0] - eps, range[1] + eps))
        self.discretizers[type_name] = hist

    def encode(self, x: torch.Tensor, type_name: str):
        # seems the same as search sorted?
        return torch.bucketize(x, self.discretizers[type_name].bin_edges, right=False, out_int32=False)

    def decode(self, x: torch.Tensor, type_name: str):
        return self.discretizers[type_name].bin_edges[x]


class BaseDataset(Dataset):
    @classmethod
    def from_config(cls, config: Config):
        return cls(**asdict(config))


class AcrossEpisodeMeta:
    def eval_setup(self, idx: int = 0):
        return self.__getitem__(idx)


class MultipleEpisodeMeta:
    datasets: list[Dataset]

    def available_fields(self, idxs: list[int] = None) -> set[str]:
        idxs = idxs or [0 for _ in range(len(self.datasets))]

        available_keys = None
        episode: EpisodeData

        for d_idx, dataset in enumerate(self.datasets):
            episode = dataset[idxs[d_idx]]
            episode_keys = set(asdict(episode).keys())

            if available_keys is None:
                available_keys = episode_keys

            available_keys = available_keys.intersection(episode_keys)

        return available_keys
