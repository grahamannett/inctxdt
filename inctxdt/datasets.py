import os
from typing import Any, Dict, List

import minari
import numpy as np
from torch.utils.data import Dataset
from minari.storage.datasets_root_dir import get_dataset_path
from inctxdt.episode_data import EpisodeData


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def _get_data_path_from_env_name(env_name: str) -> str:
    if env_name not in minari.list_local_datasets():
        minari.download_dataset(env_name)
    file_path = get_dataset_path(env_name)
    data_path = os.path.join(file_path, "data", "main_data.hdf5")
    return data_path


class MinariDataset(minari.MinariDataset):
    def __init__(self, env_name: str, seq_len: int = None):
        data_path = _get_data_path_from_env_name(env_name)
        super().__init__(data=data_path)

        self.env_name = env_name
        self.seq_len = seq_len

    def __len__(self):
        return self.total_episodes

    def __getitem__(self, idx: int):
        episode = self._data.get_episodes([self.episode_indices[idx]])[0]
        episode = self._fix_episode(episode)
        return EpisodeData(**episode)

    def _fix_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.seq_len:
            raise NotImplementedError

        if episode_data["seed"] == "None":
            episode_data["seed"] = None

        episode_data["returns_to_go"] = discounted_cumsum(episode_data["rewards"], gamma=1.0)
        episode_data["mask"] = np.ones(episode_data["total_timesteps"])

        # concatenate episode data
        episode_data["observations"] = np.concatenate(list(episode_data["observations"].values()), axis=-1)
        episode_data["env_name"] = self.env_name
        return episode_data


class AcrossEpisodeDataset(MinariDataset):
    def __init__(self, env_name: str, seq_len: int = None, max_num_epsisodes: int = 2, drop_last: bool = False):
        super().__init__(env_name, seq_len)
        self.max_num_episodes = max_num_epsisodes
        self.drop_last = drop_last

    def _get_after_episodes(self, idx: int):
        ep_idxs = self.episode_indices[idx : idx + self.max_num_episodes]
        eps = [super(AcrossEpisodeDataset, self).__getitem__(i) for i in self.ep_idxs]

    def __getitem__(self, idx: int) -> Any:
        # eps = []
        # for ep_idx in ep_idxs:
        #     # ep = super().__getitem__(ep_idx, output_cls=dict)

        ep1 = super().__getitem__(ep_idxs)

        ep = ep1.combine(ep2)

        breakpoint()
        eps = [super(AcrossEpisodeDataset, self).__getitem__(i, dict) for i in ep_idxs]

        for k, v in eps[0].items():
            eps_out[k] = np.concatenate([ep[k] for ep in eps], axis=0) if isinstance(v, (np.ndarray, int, float)) else [ep[k] for ep in eps]
        breakpoint()

        # episodes = [super().__getitem__(ep_idx, output_cls=dict) for ep_idx in ep_idxs]

        # ep_idxs = [self.ds.episode_indices[idx : idx + self.max_num_episodes]]
        episode_data = self.ds._data.get_episodes(*ep_idxs)

        to_get = self.max_num_episodes
        episodes = [super().__getitem__(idx, output_cls=dict)]

        while to_get > 0:
            if (idx := idx + 1) >= len(self):
                if self.drop_last:
                    break
                idx = 0

            episode = super().__getitem__(idx, output_cls=dict)
            episodes.append(episode)
            to_get -= 1
        return episodes


class MultipleMinariDataset(Dataset):
    """requires that the datasets share properties"""

    def __init__(self, datasets: List[MinariDataset]):
        self.datasets = datasets
        self.dataset_indices = []
        self._generate_indexes()

    def __len__(self):
        return len(self.dataset_indices)

    def __getitem__(self, index) -> Any:
        ds_idx, ep_idx = self.dataset_indices[index]
        return self.datasets[ds_idx][ep_idx]

    def _generate_indexes(self):
        for i, ds in enumerate(self.datasets):
            self.dataset_indices.extend([(i, j) for j in range(len(ds))])

    def _validate_datasets(self):
        samples = [ds[0] for ds in self.datasets]
        for i, sample in enumerate(samples):
            pass
