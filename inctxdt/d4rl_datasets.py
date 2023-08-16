import random
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple, Union

import gym
import d4rl  # put import for d4rl after gym.  its required
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from tqdm import trange

from inctxdt.env_helper import _envs_registered
from inctxdt.episode_data import EpisodeData
from inctxdt.batch import Batch, EpisodeList
from inctxdt.datasets_meta import AcrossEpisodeMeta, MultipleEpisodeMeta


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


# some utils functionalities specific for Decision Transformer
def pad_along_axis(arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def load_d4rl_trajectories(
    dataset_name: str, gamma: float = 1.0
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    dataset = gym.make(dataset_name).get_dataset()
    traj, traj_len = [], []

    data_ = defaultdict(list)
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])

        if dataset["terminals"][i] or dataset["timeouts"][i]:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(episode_data["rewards"], gamma=gamma)
            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])
            # reset trajectory buffer
            data_ = defaultdict(list)

    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": dataset["observations"].mean(0, keepdims=True),
        "obs_std": dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    return traj, info


class BaseD4RLDataset(Dataset):
    _dataset_type = "d4rl"

    def __init__(self, dataset_name: str, seq_len: int = 20, reward_scale: float = 1.0, *args, **kwargs):
        self.dataset_name = dataset_name
        self.dataset, info = load_d4rl_trajectories(dataset_name, gamma=1.0)
        self.info = info
        self.reward_scale = reward_scale
        self.seq_len = seq_len

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

        self.register_env()

    @property
    def env_name(self):
        return self.dataset_name

    def register_env(self):
        env = self.recover_environment()
        action_space = env.action_space.shape[0]
        state_space = env.observation_space.shape[0]
        _envs_registered[self.dataset_name] = {"action_space": action_space, "state_space": state_space}

    def _prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        states = traj["observations"][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        returns = traj["returns"][start_idx : start_idx + self.seq_len]
        rewards = traj["rewards"][start_idx : start_idx + self.seq_len]
        timesteps = np.arange(start_idx, start_idx + self.seq_len)

        states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        # pad up to seq_len if needed
        mask = np.hstack([np.ones(states.shape[0], dtype=int), np.zeros(self.seq_len - states.shape[0], dtype=int)])

        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        return states, actions, returns, rewards, timesteps, mask

    def recover_environment(self) -> gym.Env:
        """Recover the Gymnasium environment used to create the dataset.

        Returns:
            environment: Gymnasium environment
        """
        return gym.make(self.dataset_name)

    @staticmethod
    def collate_fn(batch_first: bool = True):
        def fn(episode_list: List[EpisodeData]) -> Batch:
            eps = EpisodeList(episode_list, batch_first=batch_first)
            batch = Batch(
                states=eps.states.pad_tensor,
                actions=eps.actions.pad_tensor,
                total_timesteps=eps.total_timesteps.tensor,
                rewards=eps.rewards.pad_tensor,
                returns_to_go=eps.returns_to_go.pad_tensor,
                timesteps=eps.timesteps.pad_tensor,
                mask=eps.mask.pad_tensor,
                id=eps.id,
                seed=eps.seed,
                env_name=eps.env_name,
                batch_size=[len(eps)],
            )
            return batch

        return fn


class D4rlDataset(BaseD4RLDataset):
    def __init__(self, dataset_name: str, *args, **kwargs):
        super().__init__(dataset_name=dataset_name, *args, **kwargs)
        self.dataset_indices = []

        for i, traj in enumerate(self.dataset):
            self.dataset_indices.extend([(i, j) for j in range(len(traj["returns"]))])

    def __len__(self):
        return len(self.dataset_indices)

    def __getitem__(self, idx: int):
        traj_idx, sample_idx = self.dataset_indices[idx]
        states, actions, returns, rewards, timesteps, mask = self._prepare_sample(traj_idx, sample_idx)

        return EpisodeData(
            states=states,
            actions=actions,
            total_timesteps=self.seq_len,
            rewards=rewards,
            returns_to_go=returns,
            id=idx,
            timesteps=timesteps,
            # terminations=terminations,
            # truncations=truncations,
            mask=mask,
            env_name=self.dataset_name,
        )


class D4rlAcrossEpisodeDataset(AcrossEpisodeMeta, D4rlDataset):
    def __init__(self, dataset_name: str, max_num_epsisodes: int = 2, *args, **kwargs):
        super().__init__(dataset_name, *args, **kwargs)
        self.max_num_episodes = max_num_epsisodes

    def __getitem__(self, idx: int) -> Any:
        idxs = [idx]

        while len(idxs) < self.max_num_episodes:
            # print("getting")
            if (new_idx := random.randint(0, len(self.dataset_indices) - 1)) not in idxs:
                idxs.append(new_idx)

        eps = [super(D4rlAcrossEpisodeDataset, self).__getitem__(i) for i in idxs]
        return EpisodeData.combine(eps)


class D4rlMultipleDataset(MultipleEpisodeMeta, D4rlDataset):
    def __init__(self, datasets: list[D4rlDataset], *args, **kwargs):
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


class IterableD4rlDataset(BaseD4RLDataset, IterableDataset):
    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
            yield self._prepare_sample(traj_idx, start_idx)


if __name__ == "__main__":
    dataset_name = "halfcheetah-medium-v2"
    seq_len = 20
    reward_scale = 0.001
    dataset = D4rlDataset(dataset_name, seq_len=seq_len, reward_scale=reward_scale)
