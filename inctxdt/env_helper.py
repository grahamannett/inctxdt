import gymnasium
import numpy as np


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


def get_env_gymnasium(env_name: str, config=None, venv: bool = True):
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
        env = gymnasium.wrappers.TransformObservation(env, lambda obs: (obs - config.state_mean) / config.state_std)
        env = gymnasium.wrappers.TransformReward(env, lambda rew: rew * config.reward_scale)
        return env

    base_env = fn()
    obs_space = base_env.observation_space

    venv = (
        gymnasium.vector.SyncVectorEnv([fn for _ in range(config.eval_episodes)], observation_space=obs_space)
        if venv
        else None
    )

    return fn, base_env, venv, obs_space, base_env.action_space


def get_env_gym(env_name: str, config=None, venv: bool = True):
    import gym
    import d4rl

    base_env = gym.make(env_name)
    base_obs = base_env.reset()
    if isinstance(base_obs, (list, tuple)):
        base_obs = base_obs[0]

    class NormalizeObservation(gym.ObservationWrapper):
        def __init__(self, env, obs_shape, mean, std):
            super().__init__(env)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float64)
            self.mean = mean.flatten()
            self.std = std.flatten()

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
        # not sure exactly but seemed like TransformObservation messed up stuff.  could be one of the other bugs though.
        env = NormalizeObservation(
            env, base_obs.shape, mean=config.state_mean.flatten(), std=config.state_std.flatten()
        )
        env = NormalizeReward(env, config.reward_scale)

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


def get_env(config, dataset=None, env_name=None, dataset_type=None):
    env_name = env_name or getattr(dataset, "env_name", config.env_name)

    dataset_type = dataset_type or getattr(dataset, "_dataset_type", config.dataset_type)

    dataset_type = dataset_type.split("_")[0]  # might have _across or _multiple

    env_fn = _fn[dataset_type]
    return env_fn(env_name, config)


#  probably need to remove
_envs_registered = {}


def _get_env_spec(env_name: str = None, dataset_name: str = None) -> tuple[int, int]:
    assert env_name or dataset_name, "Must pass in either env_name or dataset_name"
    if env_name in _envs_registered:
        return _envs_registered[env_name]["action_space"], _envs_registered[env_name]["state_space"]

    if dataset_name in _envs_registered:
        return _envs_registered[dataset_name]["action_space"], _envs_registered[dataset_name]["state_space"]

    assert False, f"env_name: {env_name} or dataset_name: {dataset_name} not found in registered envs"
