import gymnasium
import numpy as np

from inctxdt.config import Config


class FlattenEnv(gymnasium.ObservationWrapper):
    def __init__(self, env, obs_shape):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=obs_shape)

    def observation(self, obs):
        return obs["observation"]


class NormalizeObservation(gymnasium.ObservationWrapper):
    def __init__(self, env, obs_shape, mean, std):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float64)
        self.mean = mean.flatten()
        self.std = std.flatten()

    def observation(self, observation: np.ndarray) -> np.ndarray:
        out = (observation - self.mean) / self.std
        return out


def get_env_gymnasium(env_name: str, config=None, venv: bool = True, **kwargs):
    mean = kwargs.get("mean", config.state_mean)
    std = kwargs.get("std", config.state_std)
    reward_scale = kwargs.get("reward_scale", config.reward_scale)

    base_env = gymnasium.make(env_name)
    base_obs = base_env.reset()[0]

    observation_shape = base_env.observation_space

    needs_flatten = False
    if isinstance(base_obs, dict):
        needs_flatten = True
        base_obs = base_obs["observation"]
        observation_shape = observation_shape["observation"]

    def fn():
        env = gymnasium.make(env_name)
        env = FlattenEnv(env, base_obs.shape) if needs_flatten else env
        env = gymnasium.wrappers.TransformObservation(env, lambda obs: (obs - mean) / std)
        env = gymnasium.wrappers.TransformReward(env, lambda rew: rew * reward_scale)
        return env

    base_env = fn()
    obs_space = base_env.observation_space

    venv = (
        gymnasium.vector.SyncVectorEnv([fn for _ in range(config.eval_episodes)], observation_space=obs_space)
        if venv
        else None
    )

    return fn, base_env, venv, obs_space, base_env.action_space


def get_env_gym(env_name: str, config: Config, venv: bool = True, **kwargs):
    import gym
    import d4rl

    mean = kwargs.get("mean", config.state_mean)
    std = kwargs.get("std", config.state_std)
    reward_scale = kwargs.get("reward_scale", config.reward_scale)

    base_env = gym.make(env_name)
    base_obs = base_env.reset()
    if isinstance(base_obs, (list, tuple)):
        base_obs = base_obs[0]

    class NormalizeObservation(gym.ObservationWrapper):
        def __init__(self, env, obs_shape, mean, std):
            super().__init__(env)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float64)
            self.mean, self.std = mean.flatten(), std.flatten()

        def observation(self, observation: np.ndarray) -> np.ndarray:
            out = (observation - self.mean) / self.std
            return out

    class NormalizeReward(gym.RewardWrapper):
        def __init__(self, env, scale):
            super().__init__(env)
            self.scale = scale

        def reward(self, reward: float) -> float:
            return reward * self.scale

    def fn():
        env = gym.make(env_name)
        # not sure exactly but seemed like TransformObservation messed up stuff.
        # could be one of the other bugs fixed the issue i thought was this though.
        env = NormalizeObservation(env, base_obs.shape, mean=mean, std=std)
        env = NormalizeReward(env, scale=reward_scale)

        return env

    base_env = fn()
    obs_space = base_env.observation_space

    venv = (
        gym.vector.SyncVectorEnv([fn for _ in range(config.eval_episodes)], observation_space=obs_space)
        if venv
        else None
    )

    return fn, base_env, venv, obs_space, base_env.action_space


_fn = {
    "d4rl": get_env_gym,
    "minari": get_env_gymnasium,
}


def get_env(config, dataset=None, env_name=None, dataset_type=None, **kwargs):
    env_name = env_name or getattr(dataset, "env_name", config.env_name)

    dataset_type = dataset_type or getattr(dataset, "_dataset_type", config.dataset_type)

    dataset_type = dataset_type.split("_")[0]  # might have _across or _multiple

    env_fn = _fn[dataset_type]
    return env_fn(env_name, config, **kwargs)


#  probably need to remove
_envs_registered = {}
