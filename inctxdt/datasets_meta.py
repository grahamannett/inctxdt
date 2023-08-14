from dataclasses import asdict

from inctxdt.episode_data import EpisodeData
from torch.utils.data import Dataset


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
