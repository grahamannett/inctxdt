import os
from typing import Any, Dict, List

import minari
import numpy as np
from torch.utils.data import Dataset
from minari.storage.datasets_root_dir import get_dataset_path
from inctxdt.episode_data import EpisodeData
from functools import reduce


def discounted_cumsum(x: np.ndarray, gamma: float, dtype: np.dtype | str = np.float32) -> np.ndarray:
    cumsum = np.zeros_like(x, dtype=dtype)
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
        episode = EpisodeData(**self._fix_episode(episode))
        return episode

    def _fix_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.seq_len:
            raise NotImplementedError

        episode_data["env_name"] = self.env_name

        if episode_data["seed"] == "None":
            episode_data["seed"] = None

        episode_data["returns_to_go"] = discounted_cumsum(episode_data["rewards"], gamma=1.0, dtype=np.float32)
        episode_data["mask"] = np.ones(episode_data["total_timesteps"], dtype=np.float32)
        episode_data["timesteps"] = np.arange(episode_data["total_timesteps"])

        # dont do this as i dont know if the other fields are useful or what they mean
        # concatenate episode data
        # episode_data["observations"] = np.concatenate(list(episode_data["observations"].values()), axis=-1, dtype=np.float32)

        # fix dtypes of others
        episode_data["observations"] = episode_data["observations"]["observation"][:-1].astype(np.float32)
        episode_data["actions"] = episode_data["actions"].astype(np.float32)
        episode_data["rewards"] = episode_data["rewards"].astype(np.float32)

        return episode_data


class AcrossEpisodeDataset(MinariDataset):
    def __init__(self, env_name: str, seq_len: int = None, max_num_epsisodes: int = 2, drop_last: bool = False):
        super().__init__(env_name, seq_len)
        self.max_num_episodes = max_num_epsisodes
        self.drop_last = drop_last

        # just make it so we can always cycle through the episodes
        self.possible_idxs = np.append(self.episode_indices, self.episode_indices[0 : self.max_num_episodes - 1])

    def __getitem__(self, idx: int) -> Any:
        eps = [super(AcrossEpisodeDataset, self).__getitem__(i) for i in self.possible_idxs[idx : idx + self.max_num_episodes]]
        return EpisodeData.combine(eps)


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
