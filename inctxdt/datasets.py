import os
from typing import Any, Dict, List

import minari
import numpy as np
from minari.storage.datasets_root_dir import get_dataset_path
from torch.utils.data import Dataset

from inctxdt.episode_data import EpisodeData

from inctxdt.config import _envs_registered


def discounted_cumsum(x: np.ndarray, gamma: float, dtype: np.dtype | str = np.float32) -> np.ndarray:
    cumsum = np.zeros_like(x, dtype=dtype)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def _get_data_path_from_dataset_name(dataset_name: str) -> str:
    if dataset_name not in minari.list_local_datasets():
        minari.download_dataset(dataset_name)
    file_path = get_dataset_path(dataset_name)
    data_path = os.path.join(file_path, "data", "main_data.hdf5")
    return data_path


class MinariDataset(minari.MinariDataset):
    def __init__(self, dataset_name: str, seq_len: int = None):
        data_path = _get_data_path_from_dataset_name(dataset_name)
        super().__init__(data=data_path)

        self.dataset_name = dataset_name
        self.env_name = self._data.env_spec.id

        self.seq_len = seq_len

        self.setup()

    def register_env(self):
        _env = self.recover_environment()
        action_space = _env.action_space.shape[0]

        if (observation_space := _env.observation_space).__class__.__name__ == "Dict":
            observation_space = observation_space["observation"]

        observation_space = observation_space.shape[0]

        _envs_registered[self.dataset_name] = {"action_space": action_space, "state_space": observation_space}
        _envs_registered[self.env_name] = {"action_space": action_space, "state_space": observation_space}

    def setup(self):
        self.register_env()

        def fn(e):
            if isinstance(obs := e.observations, dict):
                obs = obs["observation"]
            return obs

        observations = [fn(ep) for ep in self.iterate_episodes()]

        observations = np.concatenate(observations, axis=0)
        self.state_mean = observations.mean(0)
        self.state_std = observations.std(0)

    def __len__(self):
        return self.total_episodes

    def __getitem__(self, idx: int):
        episode = self._data.get_episodes([self.episode_indices[idx]])[0]
        fixed_episode = self._fix_episode(episode)
        episode = EpisodeData(**fixed_episode)
        return episode

    def _fix_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        n_timesteps = episode_data["total_timesteps"]

        assert n_timesteps == len(episode_data["actions"])
        assert n_timesteps == len(episode_data["rewards"])

        episode_data["env_name"] = self.env_name

        if episode_data["seed"] == "None":
            episode_data["seed"] = None

        episode_data["returns_to_go"] = discounted_cumsum(episode_data["rewards"], gamma=1.0, dtype=np.float32)
        episode_data["mask"] = np.ones(episode_data["total_timesteps"], dtype=np.float32)
        episode_data["timesteps"] = np.arange(n_timesteps)
        episode_data["states"] = episode_data.pop("observations")
        if isinstance(episode_data["states"], dict):
            episode_data["states"] = episode_data["states"]["observation"]

        # dont do this as i dont know if the other fields are useful or what they mean
        # concatenate episode data

        # fix dtypes of others

        episode_data["states"] = episode_data["states"].astype(np.float32)[:n_timesteps]
        episode_data["actions"] = episode_data["actions"].astype(np.float32)
        episode_data["rewards"] = episode_data["rewards"].astype(np.float32)

        if self.seq_len:
            for k, v in episode_data.items():
                if isinstance(v, np.ndarray):
                    episode_data[k] = v[: self.seq_len]

        return episode_data


class AcrossEpisodeDataset(MinariDataset):
    def __init__(
        self,
        dataset_name: str,
        seq_len: int = None,
        max_num_epsisodes: int = 2,
        drop_last: bool = False,
    ):
        super().__init__(dataset_name, seq_len)
        self.max_num_episodes = max_num_epsisodes
        self.drop_last = drop_last

        # just make it so we can always cycle through the episodes
        self.possible_idxs = np.append(self.episode_indices, self.episode_indices[0 : self.max_num_episodes - 1])

    def __getitem__(self, idx: int) -> Any:
        eps = [
            super(AcrossEpisodeDataset, self).__getitem__(i)
            for i in self.possible_idxs[idx : idx + self.max_num_episodes]
        ]
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

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        else:
            return getattr(self.datasets[0], attr)
